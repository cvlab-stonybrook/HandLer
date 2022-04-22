import xml.etree.ElementTree as ET
import os
import json
import sys
import pdb
from fvcore.common.file_io import PathManager
import tqdm
# coco = list()
coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []
category_set = dict()
image_set = set()

category_item_id = 0
image_id = 20180000000
annotation_id = 0

def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = 1
    return category_item_id


def addImgItem(file_name,image_id, h, w):

    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = w
    image_item['height'] = h
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id


def addAnnoItem(object_name, image_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = 0
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)

def read(dirname):
    # with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
    #     fileids = np.loadtxt(f, dtype=np.str)
    fileids = os.listdir(os.path.join(dirname, "voc_anno/"))

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "voc_anno/"))
    image_id = 0
    for fileid in tqdm.tqdm(fileids):
        fileid = '.'.join(fileid.split('.')[:-1])
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")
        if not os.path.exists(jpeg_file) or not 'VOC' in fileid:
            continue

        with open(anno_file) as f:
            tree = ET.parse(f)

        addImgItem(jpeg_file, image_id, int(tree.findall("./size/height")[0].text), int(tree.findall("./size/width")[0].text))

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls != 'hand' and cls != 'hands':
                continue
            bndbox = obj.find("bndbox")
            if not bndbox:
                continue
            bbox = [float(bndbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            bbox[2] -= bbox[0]
            bbox[3] -= bbox[1]

            addAnnoItem(None, image_id, bbox)

        image_id+=1
    addCatItem('hand')


read("/archive/mingzhen/dataset/hand_det/")
json_file = '/archive/mingzhen/dataset/hand_det/coco_anno/train.json'
json.dump(coco, open(json_file, 'w'))