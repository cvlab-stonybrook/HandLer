import numpy as np
import time, math, os, copy, json, cv2, random, pdb, sys, argparse, torch, detectron2, torchvision, threading
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils
from detectron2.structures import BoxMode
from tqdm import tqdm
from .utils import iou_distance, linear_assignment,_indices_to_matches
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from scipy import stats
import pdb, copy

def assign_keypoint(keypoint, bboxes, none=None):
    dists = list()
    keypoint = keypoint[:,1:]
    max_distance = []
    for kp in keypoint:
        dist = list()
        for bbox in bboxes:
            bbox_point = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
            distance = np.sqrt(np.square(kp[0] - bbox_point[0]) + np.square(kp[1] - bbox_point[1]))
            if len(max_distance) < len(bboxes):
                max_distance.append((bbox[3]-bbox[0]-bbox[1]+bbox[2])/2)
            dist.append(distance)
        dists.append(dist)
    dists = np.array(dists)
    if dists.shape[0] == 0:
        return np.zeros([0,3])
    indices = linear_sum_assignment(np.array(dists))
    matches = list()
    rows, cols = indices
    for row, col in zip(rows, cols):
        if dists[row, col] <= max_distance[col]:
            matches.append([row, col])
    return np.array(matches)

def assign_keypoint_greedy(keypoint, bboxes, max_distance = 100):
    if len(bboxes) == 0:
        return np.zeros([0,3])
    bboxes = copy.deepcopy(np.array(bboxes))
    sort = keypoint[:,-1].argsort()
    keypoint = keypoint[sort][::-1][:,1:-1]
    matches = []
    filter = list(range(len(bboxes)))
    for i, kp in enumerate(keypoint):
        dist = list()
        for j, bbox in enumerate(bboxes[filter]):
            bbox_point = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
            distance = np.sqrt(np.square(kp[0] - bbox_point[0]) + np.square(kp[1] - bbox_point[1]))
            dist.append(distance)
        dist = np.array(dist)
        min_idx = np.argmin(dist)
        if np.min(dist) < max_distance:
            matches.append([i, filter[min_idx]])
            filter.pop(min_idx)
        if len(filter) == 0:
            break

    if len(matches) == 0:
        return np.zeros([0,3])
    return np.array(matches)

def get_pose_from_json(poses, trk_thresh=0.05, pose_thresh=1):
    result = []
    pair = []
    idx = 0
    for pose in poses:
        if pose['score'] < trk_thresh:
            continue
        annopoints = pose['annopoints']
        for annopoint in annopoints:
            points = annopoint['point']
            if len(points) > 1:
                pair.append([idx, idx+1])
            for point in points:
                score = float(point['score'])
                track_id = point['id']
                if len(point) > 0 and score > pose_thresh:
                    result.append([track_id, point['x'], point['y'], score])
                    idx += 1
    return np.array(result), np.array(pair)

class CustomPredictor:
    def __init__(self, cfg, is_flow=True): 
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        self.is_flow = is_flow
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, inputs):
        with torch.no_grad():
            image, pre_image = inputs["image"], inputs["pre_image"]
            if self.input_format == "RGB":
                image = image[:, :, ::-1]
                pre_image = pre_image[:, :, ::-1]

            height, width = image.shape[:2]

            image = self.aug.get_transform(image).apply_image(image)
            pre_image = self.aug.get_transform(pre_image).apply_image(pre_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            pre_image = torch.as_tensor(pre_image.astype("float32").transpose(2, 0, 1))

            inputs["image"], inputs["pre_image"] = image, pre_image
            inputs["height"], inputs["width"] = height, width

            predictions = self.model([inputs])
            return predictions

def get_flow(img1, img2):
    gray1 = cv2.cvtColor(cv2.resize(img1, dsize=(64, 64), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(cv2.resize(img2, dsize=(64, 64), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2,None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow.reshape([2,64,64])

def processed_bbox(bboxes, w, h):
    xmin, ymin, xmax, ymax = bboxes
    xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax)
    return np.array([xmin, ymin, xmax, ymax])

class detect_rect:
    def __init__(self):
        self.curr_frame = 0
        self.curr_rect = np.array([0, 0, 1, 1])
        self.next_rect = np.array([0, 0, 1, 1])
        self.conf = 0
        self.id = 0
        self.pre_conf = 0

    @property
    def position(self):
        x = (self.curr_rect[0] + self.curr_rect[2]) / 2
        y = (self.curr_rect[1] + self.curr_rect[3]) / 2
        return np.array([x, y])

    @property
    def size(self):
        w = self.curr_rect[2] - self.curr_rect[0]
        h = self.curr_rect[3] - self.curr_rect[1]
        return np.array([w, h])


class tracklet:
    def __init__(self, det_rect):
        self.id = det_rect.id
        self.rect_list = [det_rect]
        self.rect_num = 1
        self.last_rect = det_rect
        self.last_frame = det_rect.curr_frame
        self.no_match_frame = 0


    def add_rect(self, det_rect):
        self.rect_list.append(det_rect)
        self.rect_num = self.rect_num + 1
        self.last_rect = det_rect
        self.last_frame = det_rect.curr_frame

    def replace_rect(self, bbox):
        self.last_rect.curr_rect = copy.deepcopy(bbox)
        # self.last_rect = det_rect

    @property
    def velocity(self):
        if (self.rect_num < 2):
            return (0, 0)
        elif (self.rect_num < 6):
            return (self.rect_list[self.rect_num - 1].position - self.rect_list[self.rect_num - 2].position) / (
                        self.rect_list[self.rect_num - 1].curr_frame - self.rect_list[self.rect_num - 2].curr_frame)
        else:
            v1 = (self.rect_list[self.rect_num - 1].position - self.rect_list[self.rect_num - 4].position) / (
                        self.rect_list[self.rect_num - 1].curr_frame - self.rect_list[self.rect_num - 4].curr_frame)
            v2 = (self.rect_list[self.rect_num - 2].position - self.rect_list[self.rect_num - 5].position) / (
                        self.rect_list[self.rect_num - 2].curr_frame - self.rect_list[self.rect_num - 5].curr_frame)
            v3 = (self.rect_list[self.rect_num - 3].position - self.rect_list[self.rect_num - 6].position) / (
                        self.rect_list[self.rect_num - 3].curr_frame - self.rect_list[self.rect_num - 6].curr_frame)
            return (v1 + v2 + v3) / 3

class pose_tracklet:
    def __init__(self, keypoints, pose_id, hand_det):
        self.wrist = [keypoints]
        self.match_hand = {}
        self.pose_id = pose_id
        self.match_hand = [hand_det]
        self.pair_hand = []

    def add_trk(self, keypoints, pose_id, hand_det):
        self.wrist.append(keypoints)
        self.pose_id = pose_id
        self.match_hand.append(hand_det)

    def add_pair(self, hand_det):
        self.pair_hand.append(hand_det)

    def get_pair_hand_id(self):
        match_hand_id = [i.id for i in self.pair_hand]
        return stats.mode(match_hand_id[-20:])[0][0]

    @property
    def get_hand_pose_tracklet(self):
        return self.match_hand

    @property
    def get_recent_hand_id(self):
        match_hand_id = [i.id for i in self.match_hand]
        return stats.mode(match_hand_id[-20:])[0][0]

    @property
    def get_length(self):
        return len(self.match_hand)

def find_tracklet_by_id(tracklets, id):

    for idx, tracklet in enumerate(tracklets):
        if tracklet.id == id:
            return tracklet, idx
    return None, -1


def get_pose(pose_dets, id):
    try:
        pose_det = pose_dets[pose_dets[:, 0] == id]
    except:
        pose_det = np.zeros([0, 4])
    return pose_det

def set_det_rect(bbox, score, pre_bbox, idx):
    x1, y1, x2, y2 = map(int, bbox)

    det_conf = float(score)
    det_rect = detect_rect()
    det_rect.curr_frame = idx
    det_rect.curr_rect = np.array([x1, y1, x2, y2])
    det_rect.next_rect = pre_bbox
    det_rect.conf = det_conf
    return det_rect

def cal_iou(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4, _ = rect2[0]
    i_w = min(x2, x4) - max(x1, x3)
    i_h = min(y2, y4) - max(y1, y3)
    if (i_w <= 0 or i_h <= 0):
        return 0
    i_s = i_w * i_h
    s_1 = (x2 - x1) * (y2 - y1)
    s_2 = (x4 - x3) * (y4 - y3)
    return float(i_s) / (s_1 + s_2 - i_s)

def cal_iou_prob(rect1, rect2, rect3):
    assert len(rect2.shape) == 2
    num_box = rect2.shape[0]
    ious = np.zeros([num_box,1])
    for i in range(num_box):
        if rect2[i][-1] > 0:
            ious[i] = cal_iou(rect1, [rect2[i]]) * (rect2[i][-1]+0.1) #* cal_iou(rect3, [rect2[i]])
        else:
            ious[i] = 0
    return max(ious[0])

def cal_dist(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    ct_1 = ((x1+x2)/2, (y1+y2)/2)
    ct_2 = ((x3 + x4) / 2, (y4 + y3) / 2)
    return math.sqrt(math.pow(ct_1[0]-ct_2[0],2) + math.pow(ct_1[1]-ct_2[1], 2))

def cal_simi(det_rect1, det_rect2):
    return cal_iou(det_rect1.next_rect, det_rect2.curr_rect)

def cal_simi_track_det(track, det_rect, prob=False):
    if (det_rect.curr_frame <= track.last_frame):
        print("cal_simi_track_det error")
        return 0
    elif (det_rect.curr_frame - track.last_frame == 1):
        if not prob:
            return cal_iou(track.last_rect.curr_rect, det_rect.next_rect)
        else:
            return cal_iou_prob(track.last_rect.curr_rect, det_rect.next_rect, det_rect.curr_rect)
    else:
        pred_rect = track.last_rect.curr_rect + np.append(track.velocity, track.velocity) * (
                    det_rect.curr_frame - track.last_frame)
        if not prob:
            return cal_iou(track.last_rect.curr_rect, det_rect.next_rect)
        else:
            return cal_iou_prob(track.last_rect.curr_rect, det_rect.next_rect, det_rect.curr_rect)

def cal_simi_track_det_l2(track, det_rect):
    if (det_rect.curr_frame <= track.last_frame):
        print("cal_simi_track_det error")
        return 0
    elif (det_rect.curr_frame - track.last_frame == 1):
        return cal_dist(track.last_rect.curr_rect, det_rect.next_rect)
    else:
        pred_rect = track.last_rect.curr_rect + np.append(track.velocity, track.velocity) * (
                    det_rect.curr_frame - track.last_frame)
        return cal_dist(pred_rect, det_rect.next_rect)

def track_det_match(tracklet_list, det_rect_list, min_iou=0.5, l2=False, prob=False):
    num1 = len(tracklet_list)
    num2 = len(det_rect_list)
    cost_mat = np.zeros((num1, num2))
    for i in range(num1):
        for j in range(num2):
            if l2:
                cost_mat[i, j] = cal_simi_track_det_l2(tracklet_list[i], det_rect_list[j])
            elif prob:
                cost_mat[i, j] = -cal_simi_track_det(tracklet_list[i], det_rect_list[j], prob=True)
            else:
                cost_mat[i, j] = -cal_simi_track_det(tracklet_list[i], det_rect_list[j])

    match_result = linear_sum_assignment(cost_mat)
    match_result = np.asarray(match_result)
    match_result = np.transpose(match_result)

    matches, unmatched1, unmatched2 = [], [], []
    for i in range(num1):
        if i not in match_result[:, 0]:
            unmatched1.append(i)
    for j in range(num2):
        if j not in match_result[:, 1]:
            unmatched2.append(j)
    for i, j in match_result:
        if cost_mat[i, j] > -min_iou and not l2:
            unmatched1.append(i)
            unmatched2.append(j)
        elif l2 and cost_mat[i, j] > 200:
            unmatched1.append(i)
            unmatched2.append(j)
        else:
            matches.append((i, j))
    return np.array(matches), np.array(unmatched1), np.array(unmatched2)


def draw_caption(image, box, caption, color):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 8), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)