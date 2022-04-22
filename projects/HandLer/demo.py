from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from handler.config import add_handler_config
from tqdm import tqdm
from scipy import stats
from tracking_utils import *
import pdb
import torch.nn.functional as F
import json, threading, sys, os, copy
setup_logger()

color_list = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (128, 0, 255),
              (0, 128, 255), (128, 255, 0), (0, 255, 128), (255, 128, 0), (255, 0, 128), (128, 128, 255),
              (128, 255, 128), (255, 128, 128), (128, 128, 0), (128, 0, 128)]

def parse_args():
    parser = argparse.ArgumentParser(description='Test hand tracking network')
    # general

    parser.add_argument('--weights',
                        required=False,
                        default='',
                        type=str)

    parser.add_argument('--cfg',
                        required=False,
                        default='configs/yt_trk.yaml',
                        type=str)

    parser.add_argument('--output',
                        required=False,
                        default='./result/res.mp4',
                        type=str)

    parser.add_argument('--input',
                        required=False,
                        default='./input.mp4',
                        type=str)

    args = parser.parse_args()

    return args

def run_each_dataset(model, dataset_path, output):
    vidcap = cv2.VideoCapture(dataset_path)

    confidence_threshold = 0.6
    IOU_threshold = 0.01
    retention_threshold = 20
    init_threshold = 0.8

    det_list_all = []
    tracklet_all = []
    max_id = 0
    idx = -1


    start_time = time.time()
    success = True
    while success:
        idx+=1
        det_list_all.append([])
        if idx % 50 == 0:
            print('current frame is '+str(idx))
        success, image = vidcap.read()
        if image is None:
            continue

        i = idx
        with torch.no_grad():
            if idx == 0:
                outputs, heatmap, trk_res, feats = model({"image": image, "pre_image": image, "pose_hm": None}) 
                image_h, image_w = image.shape[:2]
                out = cv2.VideoWriter(output,
                                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                                      (image_w, image_h))
            else:
                outputs, heatmap, trk_res, _ = model({"image": image, "pre_image": pre_img, "heatmap": heatmap, "pose_hm": None, 'pre_features': feats})

            trk_res = trk_res[0]
            outputs, trk_res = outputs[0], trk_res[0]
            bboxes = copy.deepcopy(outputs["instances"].pred_boxes.tensor.cpu().numpy())
            scores = copy.deepcopy(outputs["instances"].scores.cpu().numpy())
            pre_bboxes = copy.deepcopy(trk_res["instances"].pred_boxes.tensor.cpu().numpy())
            pre_scores = trk_res["instances"].scores.cpu().numpy()
            if len(pre_bboxes)>0 and len(pre_bboxes)==len(bboxes):
                pre_bboxes_with_scores = np.concatenate((pre_bboxes[None,:,:], pre_scores[None,:, None]), 2)
            elif len(bboxes)>0:
                pre_bboxes_with_scores = np.concatenate((bboxes, scores[None,:]), 1)
            pre_img = image


            length = bboxes.shape[0]
            for j in range(length):
                det_rect = set_det_rect(bboxes[j], scores[j], pre_bboxes_with_scores[:,j,:], idx)
                if det_rect.conf > confidence_threshold:
                    det_list_all[det_rect.curr_frame].append(det_rect)

            if i == 0:
                for j in range(len(det_list_all[i])):    
                    det_list_all[i][j].id = j + 1
                    max_id = max(max_id, j + 1)
                    track = tracklet(det_list_all[i][j])
                    tracklet_all.append(track)
                continue

            matches, unmatched1, unmatched2 = track_det_match(tracklet_all, det_list_all[i], IOU_threshold)

            matched_id_list = []
            for j in range(len(matches)):
                det_list_all[i][matches[j][1]].id = tracklet_all[matches[j][0]].id
                tracklet_all[matches[j][0]].add_rect(det_list_all[i][matches[j][1]])
                matched_id_list.append(det_list_all[i][matches[j][1]].id)
                continue



            for j in range(len(unmatched2)):
                if det_list_all[i][unmatched2[j]].conf >= init_threshold:
                    det_list_all[i][unmatched2[j]].id = max_id + 1
                    max_id = max_id + 1
                    track = tracklet(det_list_all[i][unmatched2[j]])
                    tracklet_all.append(track)

            delete_track_list = []
            for j in range(len(unmatched1)):
                tracklet_all[unmatched1[j]].no_match_frame = tracklet_all[unmatched1[j]].no_match_frame + 1
                if (tracklet_all[unmatched1[j]].no_match_frame >= retention_threshold):
                    delete_track_list.append(unmatched1[j])

            origin_index = set([k for k in range(len(tracklet_all))])
            delete_index = set(delete_track_list)
            left_index = list(origin_index - delete_index)
            tracklet_all = [tracklet_all[k] for k in left_index]
    # **************visualize tracking result**************  **
        for j in range(len(det_list_all[-1])):
            x1, y1, x2, y2 = det_list_all[i][j].curr_rect.astype(int)
            trace_id = det_list_all[i][j].id
            if trace_id == 0:
                continue

            cv2.rectangle(image, (x1, y1), (x2, y2),
                          color_list[trace_id % len(color_list)], 5)

            cv2.putText(image, str(int(trace_id)), (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        out.write(image)

def main():
    args = parse_args()
    cfg = get_cfg()
    add_handler_config(cfg)
    cfg.merge_from_file(args.cfg)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model 
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    print("load weights from " + cfg.MODEL.WEIGHTS)
    model = CustomPredictor(cfg)
    run_each_dataset(model, args.input, args.output)


if __name__ == '__main__':
    main()