import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch
import cv2
import time
from detectron2.config import configurable
import random
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]
def get_flow(img1, img2):
    gray1 = cv2.cvtColor(cv2.resize(img1, dsize=(64, 64), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(cv2.resize(img2, dsize=(64, 64), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2,None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.
    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.
    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = augmentations
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        self.offset_scale           = 1
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }
        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def __call__(self, dataset_dict):
        seed = int(time.time())
        random.seed(seed)
        torch.manual_seed(seed)
        # dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        # utils.check_image_size(dataset_dict, image)

        aug_input = T.StandardAugInput(image)
        transforms = aug_input.apply_augmentations(self.augmentations)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        pre_image = utils.read_image(dataset_dict["pre_file_name"], format=self.image_format)
        # utils.check_image_size(dataset_dict, pre_image)
        aug_pre_input = T.StandardAugInput(pre_image)
        pre_ransforms = aug_pre_input.apply_augmentations(self.augmentations)
        pre_image = aug_pre_input.image
        dataset_dict["pre_image"] = torch.as_tensor(np.ascontiguousarray(pre_image.transpose(2, 0, 1)))
        if 'pre_hm_path' in dataset_dict:
            pose_hm = torch.as_tensor((np.load(dataset_dict['hm_path'])))
            dataset_dict['pose_hm'] = torch.nn.functional.interpolate(pose_hm, image_shape).squeeze()
            pre_pose_hm = torch.as_tensor((np.load(dataset_dict['pre_hm_path'])))
            dataset_dict['pre_pose_hm'] = torch.nn.functional.interpolate(pre_pose_hm, image_shape).squeeze()

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
            ]

            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

            pre_annos = [
                utils.transform_instance_annotations(
                    obj, pre_ransforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("pre_annotations")
            ]
            pre_instances = utils.annotations_to_instances(
                pre_annos, image_shape, mask_format=self.instance_mask_format
            )
            # dataset_dict["pre_instances"] = utils.filter_empty_instances(pre_instances)
            dataset_dict["pre_instances"] = pre_instances
            dataset_dict["pre_loc"], dataset_dict["cur_loc"] = self.get_offset(dataset_dict["pre_instances"], dataset_dict["instances"])

        return dataset_dict

    def get_offset(self, pre, cur):
        ins_length = len(cur)
        if ins_length < 1:
            return None
        height = int(cur._image_size[1] / self.offset_scale)
        width = int(cur._image_size[0] / self.offset_scale)
        pre_loc = torch.zeros([ins_length, width, height])
        cur_loc = torch.zeros([ins_length, width, height])
        for i in range(ins_length):
            if pre[i].gt_classes == 1:
                continue

            x1, y1, x2, y2 = (pre[i].gt_boxes.tensor.numpy()[0] / self.offset_scale).astype(int)
            x3, y3, x4, y4 = (cur[i].gt_boxes.tensor.numpy().astype(int)[0] / self.offset_scale).astype(int)

            area_ratio = ((x2 - x1) * (y2 - y1)) / ((x4 - x3) * (y4 - y3))

            if area_ratio < 1:
                pre_loc[i][x1:x2, y1:y2] = 1
                cur_loc[i][x3:x4, y3:y4] = area_ratio
            else:
                pre_loc[i][x1:x2, y1:y2] = 1 / area_ratio
                cur_loc[i][x3:x4, y3:y4] = 1
        return torch.as_tensor(pre_loc).unsqueeze(0), torch.as_tensor(cur_loc).unsqueeze(0)


