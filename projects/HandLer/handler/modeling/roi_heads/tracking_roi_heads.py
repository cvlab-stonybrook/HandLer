# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import json
import math
import torch
from torch import nn
import copy
import pdb
from torch.autograd.function import Function
from typing import Dict, List, Optional, Tuple, Union

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from .custom_fast_rcnn import trk_inference
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference, FastRCNNOutputLayers
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads
from detectron2.modeling.roi_heads.box_head import build_box_head
from .custom_fast_rcnn import CustomFastRCNNOutputLayers
from detectron2.structures import Boxes
@ROI_HEADS_REGISTRY.register()
class TrackingCascadeROIHeads(CascadeROIHeads): 

    @configurable
    def __init__(
            self, *, box_in_features, box_pooler, box_heads, box_predictors, proposal_matchers,
            trk_predictor, trk_pooler, trk_head, trk_refine,  **kwargs
    ):
        super().__init__(box_in_features=box_in_features, box_pooler=box_pooler, box_heads=box_heads,
                         box_predictors=box_predictors, proposal_matchers=proposal_matchers, **kwargs)
        self.trk_predictor = trk_predictor
        self.trk_pooler = trk_pooler
        self.trk_head = trk_head
        self.trk_refine = trk_refine


    @classmethod
    def _init_box_head(self, cfg, input_shape):
        self.mult_proposal_score = cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        ret = super()._init_box_head(cfg, input_shape)

        in_channels = [input_shape[f].channels for f in in_features][0]

        del ret['box_predictors']
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        box_predictors = []
        for box_head, bbox_reg_weights in zip(ret['box_heads'], cascade_bbox_reg_weights):
            box_predictors.append(
                CustomFastRCNNOutputLayers(
                    cfg, box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights)
                ))
        ret['box_predictors'] = box_predictors

        trk_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        trk_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        trk_refine = build_box_head(
            cfg, ShapeSpec(channels=in_channels+2, height=pooler_resolution, width=pooler_resolution)
        )
        trk_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        ret['trk_predictor'] = trk_predictor
        ret['trk_head'] = trk_head
        ret['trk_refine'] = trk_refine
        ret['trk_pooler'] = trk_pooler

        self.debug = cfg.DEBUG
        if self.debug:
            self.debug_show_name = cfg.DEBUG_SHOW_NAME
            self.save_debug = cfg.SAVE_DEBUG
            self.vis_thresh = cfg.VIS_THRESH
            self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
            self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
        return ret


    def _forward_box(self, features, proposals, targets=None):
        """
        Add mult proposal scores at testing
        """
        if (not self.training) and self.mult_proposal_score:
            if len(proposals) > 0 and proposals[0].has('scores'):
                proposal_scores = [
                    p.get('scores') for p in proposals]
            else:
                proposal_scores = [
                    p.get('objectness_logits') for p in proposals]
        
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]
        for k in range(self.num_cascade_stages):
            if k > 0:
                proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
                if self.training:
                    proposals, matched_idxs = self._match_and_label_boxes(proposals, k, targets)
            predictions = self._run_stage(features, proposals, k)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
            head_outputs.append((self.box_predictor[k], predictions, proposals))

        if self.training:
            losses = {}
            storage = get_event_storage()
            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                with storage.name_scope("stage{}".format(stage)):
                    stage_losses = predictor.losses(predictions, proposals)
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
            return losses, matched_idxs, proposals
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            
            if self.mult_proposal_score:
                scores = [(s * ps[:, None]) ** 0.5 \
                    for s, ps in zip(scores, proposal_scores)]

            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes(predictions, proposals)
            pred_instances, matched_idxs = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )
            
            return pred_instances, matched_idxs

    def relocating_proposals(self, proposals, index_scale=None):
        if index_scale is None:
            return proposals
        proposals = copy.deepcopy(proposals)
        index, scale = index_scale
        for proposal in proposals:
            proposal_boxes = proposal.proposal_boxes
            proposal_center = proposal_boxes.get_centers()
            proposal_boxes_tensor = proposal_boxes.tensor
            proposal_boxes_tensor[:,index] -= (proposal_center[:,index] - proposal_boxes_tensor[:,index]) * scale
            proposal_boxes_tensor[:,index+2] -= (proposal_center[:,index] - proposal_boxes_tensor[:,index+2]) * scale
            proposal.proposal_boxes = Boxes(proposal_boxes_tensor)
        return proposals

    def _forward_tracking(
            self, features, proposals, matched_idxs, offsets, targets=None
    ):
        alpha = 0.5
        beta = 0.08
        trk_losses = {}
        if self.training:
            for matched_idx, proposals_per_image, targets_per_image in zip(matched_idxs, proposals, targets):
                if len(targets_per_image) > 0:
                    gt_boxes = targets_per_image.gt_boxes[matched_idx]
                    gt_classes = targets_per_image.gt_classes[matched_idx] + proposals_per_image.gt_classes
                    gt_classes[gt_classes>1] = 1
                else:
                    gt_boxes = Boxes(
                        targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                    )
                    gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                proposals_per_image.gt_boxes = gt_boxes
                proposals_per_image.gt_classes = gt_classes
        else:
            pred_instances_list = []
        
        relocated_list = [
            # [0, beta], [0, -beta], [1, beta], [1, -beta], 
            None
        ]
        ###### adding detached here!!!!!
        ### ablation
        concat_features = [torch.cat((feature, offset), dim=1) for feature, offset in zip(features, offsets)]
        for idx, relocated_index in enumerate(relocated_list):
            relocated_proposals = self.relocating_proposals(proposals, relocated_index)
            for i, trk_head in enumerate([self.trk_refine, self.trk_head]):
                if i == 0:
                    box_features = self.trk_pooler(concat_features, [x.proposal_boxes for x in relocated_proposals])
                    box_features = trk_head(box_features)
                    predictions = self.trk_predictor(box_features)
                    boxes = self.trk_predictor.predict_boxes(predictions, relocated_proposals)
                    scores = self.trk_predictor.predict_probs(predictions, relocated_proposals)
                    for proposals_per_image, boxes_per_image, scores_per_image in zip(relocated_proposals, boxes, scores):
                        proposals_per_image.proposal_boxes = Boxes(boxes_per_image.detach())
                        proposals_per_image.s = scores_per_image[:,0].detach()
                else:
                    box_features = self.trk_pooler(list(features), [x.proposal_boxes for x in relocated_proposals])
                    box_features = trk_head(box_features)
                    predictions = self.trk_predictor(box_features)
            del box_features

            if self.training:
                trk_losses_per_stage = self.trk_predictor.losses(predictions, relocated_proposals)
                trk_losses['loss_cls_trk_stage{}'.format(idx)] = trk_losses_per_stage.pop('loss_cls') * alpha / len(relocated_list)
                trk_losses['loss_box_reg_trk_stage{}'.format(idx)] = trk_losses_per_stage.pop('loss_box_reg') * alpha / len(relocated_list)   
            else:
                boxes = self.trk_predictor.predict_boxes(predictions, relocated_proposals)
                scores = self.trk_predictor.predict_probs(predictions, relocated_proposals)
                image_shapes = [x.image_size for x in proposals]

                pred_instances = []
                for image_shape, box, score, matched_idx in zip(image_shapes, boxes, scores, matched_idxs):
                    box, score = Boxes(box[matched_idx]), score[matched_idx]
                    box.clip(image_shape)
                    pred_instance = Instances(image_shape)
                    pred_instance.pred_boxes = box
                    pred_instance.scores = score[:, 0]
                    pred_instance.pred_classes = score[:, 1]
                    pred_instances.append(pred_instance)
                pred_instances_list.append(pred_instances)
        if self.training:
            return trk_losses
        else:
            return pred_instances_list

    def forward(self, images, features, proposals, offset_feats, offset, targets=None, pre_target=None):
        if not self.debug:
            del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        cur_features = [features[f] for f in self.box_in_features]
        if self.training:
            losses, matched_idxs, final_proposals = self._forward_box(cur_features, proposals, targets)
            losses.update(self._forward_mask(cur_features, proposals))
            losses.update(self._forward_keypoint(cur_features, proposals))
            trk_losses = self._forward_tracking(offset_feats, final_proposals, matched_idxs, offset, pre_target)
            losses.update(trk_losses)
            return proposals, losses
        else:
            pred_instances, matched_idxs = self._forward_box(cur_features, proposals)
            pred_instances = self.forward_with_given_boxes(cur_features, pred_instances)
            trk_instances = self._forward_tracking(offset_feats, proposals, matched_idxs, offset)
            assert len(pred_instances[0]) == len(trk_instances[0][0])
            if self.debug:
                from ..debug import debug_second_stage
                denormalizer = lambda x: x * self.pixel_std + self.pixel_mean
                debug_second_stage(
                    [denormalizer(x.clone()) for x in images],
                    pred_instances, proposals=proposals,
                    save_debug=self.save_debug,
                    debug_show_name=self.debug_show_name,
                    vis_thresh=self.vis_thresh)
            return pred_instances, trk_instances

    # def refine_proposals_w_offset(self, proposals, offsets):
    #     ## refine proposals by offset
    #     import pdb; pdb.set_trace()
    #     query_range = 2
    #     offset_h, offset_w = offsets[0].shape[-2:]
    #     for proposals_per_image, offset in zip(proposals, offsets):
    #         img_h, img_w = proposals_per_image.image_size
    #         proposal_boxes = proposals_per_image.proposal_boxes.tensor
    #         for x, y in proposal_boxes.view([2, 2]):
    #             ratio_x = x/img_w*offset_w
    #             ratio_y = y/img_h*offset_h

            # pass

    @torch.no_grad()
    def _match_and_label_boxes(self, proposals, stage, targets):
        num_fg_samples, num_bg_samples = [], []
        matched_idxs_list = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)
            matched_idxs_list.append(matched_idxs)
            if len(targets_per_image) > 0:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                gt_classes[proposal_labels == 0] = self.num_classes
                gt_boxes = targets_per_image.gt_boxes[matched_idxs]
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_boxes = gt_boxes

            num_fg_samples.append((proposal_labels == 1).sum().item())
            num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])

        storage = get_event_storage()
        storage.put_scalar(
            "stage{}/roi_head/num_fg_samples".format(stage),
            sum(num_fg_samples) / len(num_fg_samples),
        )
        storage.put_scalar(
            "stage{}/roi_head/num_bg_samples".format(stage),
            sum(num_bg_samples) / len(num_bg_samples),
        )

        return proposals, matched_idxs_list


