# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from typing import Optional, Tuple
import torch
import copy
import pdb
from torch import nn
import torch.nn.functional as F

from ..backbone.flownet import offset_loss, pose_offset_loss

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.structures import Instances
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
import pdb
from detectron2.data import detection_utils as utils

__all__ = ["GeneralizedRCNN_siamese"]

@META_ARCH_REGISTRY.register()
class GeneralizedRCNN_siamese(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        super().__init__(backbone=backbone,proposal_generator=proposal_generator,roi_heads=roi_heads,
                         pixel_mean=pixel_mean,pixel_std=pixel_std,input_format=input_format,vis_period=vis_period)

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        losses = {}
        # bs = len(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        pre_images = self.preprocess_pre_image(batched_inputs)
        if "instances" in batched_inputs[0]: 
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        if "pre_instances" in batched_inputs[0]:
            pre_gt_instances = [x["pre_instances"].to(self.device) for x in batched_inputs]
        else:
            pre_gt_instances = None
        if 'pose_hm' in batched_inputs[0]:
            pose_hm = [x["pose_hm"].to(self.device) for x in batched_inputs]
            pose_hm = ImageList.from_tensors(pose_hm, self.backbone.size_divisibility).tensor
            pre_pose_hm = [x["pre_pose_hm"].to(self.device) for x in batched_inputs]
            pre_pose_hm = ImageList.from_tensors(pre_pose_hm, self.backbone.size_divisibility).tensor
        else:
            pose_hm = None

        if "cur_loc" in batched_inputs[0]:
            cur_loc = [x["cur_loc"].to(self.device) for x in batched_inputs]
            pre_loc = [x["pre_loc"].to(self.device) for x in batched_inputs]

        pre_input_dict = {'cur': pre_images.tensor}
        pre_features, pre_ori_feats, _, _, _ = self.backbone(pre_input_dict)
        _, pre_proposal_losses, heatmap = self.proposal_generator(pre_images, pre_features, pre_gt_instances, return_hm=True)

        input_dict = {'cur': images.tensor, 'pre_ori_feats': pre_ori_feats, 'heatmap': heatmap, 
                        'pre_fpn_feats': pre_features}
        features, _, offset, offset_feats, offset_hm = self.backbone(input_dict)

        offset_losses = offset_loss(offset, cur_loc, pre_loc)

        # if pose_hm is not None:
        #     offsets_size = offset[0].shape[-2:]
        #     pose_hm=F.interpolate(pose_hm, offsets_size)
        #     pre_pose_hm=F.interpolate(pre_pose_hm, offsets_size)
        #     pred_hm = F.grid_sample(pre_pose_hm, offset[0].permute(0, 2, 3, 1), mode="bilinear", padding_mode="border", align_corners=True)
        #     pose_offset_losses = pose_offset_loss(offset_loss, pred_hm, pose_hm)
        #     losses.update(pose_offset_losses)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, offset_feats, offset, gt_instances, pre_gt_instances)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        
        losses.update(detector_losses)
        losses.update(offset_losses)
        losses.update(proposal_losses)
        return losses


    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        pre_images = self.preprocess_pre_image(batched_inputs)
        # pdb.set_trace()

        if True:# 'pre_features' not in batched_inputs[0]:
            pre_input_dict = {'cur': pre_images.tensor.float()}
            pre_features, pre_ori_feats, _, _, _ = self.backbone(pre_input_dict)
        else:
            pre_features, pre_ori_feats = batched_inputs[0]['pre_features']

        if 'heatmap' in batched_inputs[0]:
            heatmap = batched_inputs[0]["heatmap"]
        else:
            # pdb.set_trace()
            _, _, heatmap = self.proposal_generator(pre_images, pre_features, None, return_hm=True)

        # if 'pose_hm' in batched_inputs[0]:
        #     pose_hm = [x["pose_hm"].to(self.device) for x in batched_inputs]
        #     pose_hm = ImageList.from_tensors(pose_hm, self.backbone.size_divisibility).tensor
        # else:
        #     pose_hm = None

        # pdb.set_trace()
        input_dict = {'cur': images.tensor, 'pre_ori_feats': pre_ori_feats, 'heatmap': heatmap, 'pre_fpn_feats': pre_features}
        features, ori_feats, offset, offset_feats, offset_hm = self.backbone(input_dict)
        if detected_instances is None:
            if self.proposal_generator:
                proposals, _, heatmap = self.proposal_generator(images, features, None, return_hm=True)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, trk_results = self.roi_heads(images, features, proposals, offset_feats, offset)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GeneralizedRCNN_siamese._postprocess(results, batched_inputs, images.image_sizes), heatmap, \
                   [GeneralizedRCNN_siamese._postprocess(trk_res, batched_inputs, images.image_sizes) for trk_res in trk_results], [features, ori_feats]
        else:
            return results

    def preprocess_pre_image(self, batched_inputs):
        images = [x["pre_image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    # @staticmethod
    # def _postprocess(instances, batched_inputs, image_sizes):
    #     """
    #     Rescale the output instances to the target size.
    #     """
    #     # note: private function; subject to changes
    #     processed_results = []
    #     for results, input_per_image, image_size in zip(
    #             instances, batched_inputs, image_sizes
    #     ):
    #         output_height = input_per_image.get("height", image_size[0])
    #         output_width = input_per_image.get("width", image_size[1])
    #         # r = detector_postprocess(results_per_image, height, width, filter)
    #
    #         if isinstance(output_height, torch.Tensor):
    #             output_width_tmp = output_width.float()
    #             output_height_tmp = output_height.float()
    #             new_size = torch.stack([output_height, output_width])
    #         else:
    #             new_size = (output_height, output_width)
    #             output_width_tmp = output_width
    #             output_height_tmp = output_height
    #
    #         scale_x, scale_y = (
    #             output_width_tmp / results.image_size[1],
    #             output_height_tmp / results.image_size[0],
    #         )
    #         results = Instances(new_size, **results.get_fields())
    #
    #         if results.has("pred_boxes"):
    #             output_boxes = results.pred_boxes
    #         elif results.has("proposal_boxes"):
    #             output_boxes = results.proposal_boxes
    #
    #         output_boxes.scale(scale_x, scale_y)
    #
    #         output_boxes.clip(results.image_size)
    #
    #         processed_results.append({"instances": results})
    #     return processed_results

