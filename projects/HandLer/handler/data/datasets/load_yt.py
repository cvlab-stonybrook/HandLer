import os
import cv2
import copy
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import pycocotools.coco as pycoco
import pdb
import random
__all__ = ["register_coco_instances"]

# fmt: off
# CLASS_NAMES = ["hand"]
# fmt: on

def get_hand_dicts(img_dir: str, split: str):

    ann_path = os.path.join(img_dir, 'annotations', split+".json")
    print(ann_path)
    coco = pycoco.COCO(ann_path)
    images = coco.getImgIds()

    dataset_dicts = []
    for img_id in images:
        img_info = coco.loadImgs(ids=[img_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        annos = copy.deepcopy(coco.loadAnns(ids=ann_ids))
        record = {}
        filename = os.path.join(img_dir, img_info["file_name"])

        height, width = img_info['height'], img_info['width']
        frame_id = img_info['frame_id']
        record['file_name'] = filename
        record["image_id"] = img_id
        record["height"] = height
        record["width"] = width
        record['hm_path'] = filename.replace('images', 'pose_hm').replace('jpg', 'npy')

        objs = {}
        for anno in annos:
            xmin, ymin, w, h = anno['bbox']
            ymax = ymin + h
            xmax = xmin + w
            obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
            }
            objs[anno["track_id"]] = obj

        if img_info['prev_image_id'] == -1:
            record["pre_file_name"] = filename
            record["annotations"] = copy.deepcopy(list(objs.values()))
            record["pre_annotations"] = copy.deepcopy(list(objs.values()))
            record['pre_hm_path'] = record['hm_path']
        else:
            empty_obj = {
                "bbox": [0, 0, 1, 1],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 1,
            }
            comb_objs = []
            record["annotations"] = copy.deepcopy(list(objs.values()))
            if frame_id > 1:
                offset = random.randint(1, min(5, frame_id-1))
            try:
                last = copy.deepcopy(dataset_dicts[-offset])
            except:
                print(offset)
                print(len(dataset_dicts))
            record["pre_file_name"] = last["file_name"]
            record["pre_hm"] = last['annotations']
            record['pre_hm_path'] = last['hm_path'] 
            ann_ids = coco.getAnnIds(imgIds=[last['image_id']])
            pre_annos = coco.loadAnns(ann_ids)

            pre_objs = {}
            for pre_anno in pre_annos:
                xmin, ymin, w, h = pre_anno['bbox']
                ymax = ymin + h
                xmax = xmin + w

                obj = {
                    "bbox": [xmin, ymin, xmax, ymax],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0,
                }
                pre_objs[pre_anno["track_id"]] = obj

            for k, v in objs.items():
                comb_objs.append(copy.deepcopy(pre_objs).get(k, copy.deepcopy(empty_obj)))

            record["pre_annotations"] = comb_objs

            assert len(record["pre_annotations"]) == len(record["annotations"])

        dataset_dicts.append(record)

    return dataset_dicts


def register_coco_instances(name, image_root, split):
    DatasetCatalog.register(name, lambda: get_hand_dicts(image_root, split))
    json_file = os.path.join(dirname, 'annotations', split + ".json")
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco"
    )

splits = ["train", "test"]
dirname = "/nfs/bigneuron/mingzhen/dataset/Youtube-Hands"
for split in splits:
    register_coco_instances("hands_" + split,  dirname, split)
