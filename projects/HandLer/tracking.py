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


def parse_args():
    parser = argparse.ArgumentParser(description='Test hand tracking network')
    # general

    parser.add_argument('--weights',
                        required=False,
                        default='./model_best.pth',
                        type=str)

    parser.add_argument('--cfg',
                        required=False,
                        default='configs/yt_trk.yaml',
                        type=str)

    parser.add_argument('--pose_assocaition',
                        required=False,
                        default=False,
                        action='store_true')

    parser.add_argument('--root_path',
                        required=False,
                        default='',
                        type=str)

    parser.add_argument('--out_dir',
                        required=False,
                        default='./result/',
                        type=str)

    args = parser.parse_args()

    return args

def run_each_dataset(args, model, dataset_path, subset, vid_name, out_dir):

    img_list = os.listdir(os.path.join(dataset_path, subset, vid_name, 'img1'))
    img_list = [os.path.join(dataset_path, subset, vid_name, 'img1', _) for _ in img_list if
                ('jpg' in _) or ('png' in _)]
    img_list = sorted(img_list)
    img_len = len(img_list)

    t = tqdm(range(img_len))
    t.set_description(vid_name)
    pose_tracklets = {}
    det_list_all = []
    tracklet_all = []
    max_id = 0
    for i in range(img_len):
        det_list_all.append([])

    interploate_gap = 0
    rematch_threshold = 0.1
    confidence_threshold = 0.6
    IOU_threshold = 0.01
    if 'mpii' in vid_name:
        retention_threshold = 50
        init_threshold = 0.9
    else:
        retention_threshold = 100
        init_threshold = 0.85

    pose_dets_file = os.path.join(dataset_path, subset, vid_name, 'det/posetrack.json')
    with open(pose_dets_file,'r') as f:
        pose_dets = json.load(f)

    

    start_time = time.time()
    for idx in t:
        i = idx
        with torch.no_grad():
            data_path1 = img_list[min(idx, img_len - 1)]
            im1 = cv2.imread(os.path.join(data_path1))
            h, w = im1.shape[:2]
            if idx == 0:
                outputs, heatmap, trk_results, feats = model({"image": im1, "pre_image": im1}) 
            else:
                data_path2 = img_list[min(idx - 1, img_len - 1)]
                im2 = cv2.imread(os.path.join(data_path2))
                outputs, heatmap, trk_results, feats = model({"image": im1, "pre_image": im2, "heatmap": heatmap, 'pre_features': feats})


            outputs = outputs[0]
            bboxes = copy.deepcopy(outputs["instances"].pred_boxes.tensor.cpu().numpy())
            scores = copy.deepcopy(outputs["instances"].scores.cpu().numpy())
            num_box = bboxes.shape[0]

            pre_bboxes, pre_scores = [], []
            for trk_res in trk_results:
                pre_box = trk_res[0]["instances"].pred_boxes.tensor.cpu().numpy()
                pre_score = trk_res[0]["instances"].scores.cpu().numpy()
                pre_box_per_stage = np.zeros([num_box, 4])
                pre_score_per_stage = np.zeros([num_box])
                for j in range(min(len(pre_box), num_box)):
                    pre_box_per_stage[j] = pre_box[j] 
                    pre_score_per_stage[j] = pre_score[j]
                pre_bboxes.append(pre_box_per_stage)
                pre_scores.append(np.expand_dims(pre_score_per_stage,1))
            pre_bboxes = np.transpose(np.array(pre_bboxes), (1,0,2))
            pre_scores = np.transpose(np.array(pre_scores), (1,0,2))
            pre_bboxes_with_scores = np.concatenate((pre_bboxes, pre_scores), 2)
            frame_id = np.ones((bboxes.shape[0], 1))*idx
            scores = np.expand_dims(scores, 1)

            length = bboxes.shape[0]
            low_score_det = []
            for j in range(length):
                det_rect = set_det_rect(bboxes[j], scores[j], pre_bboxes_with_scores[j], idx)
                if det_rect.conf > confidence_threshold:
                    det_list_all[i].append(det_rect)
                else:
                    low_score_det.append(det_rect)

            ################################# pose assciation  ############################################ 
            if args.pose_assocaition:
                pose_det = pose_dets.get(str(i+1), [])
                cur_pose_tracklet = {}
                cur_hands_paris = {}
                trks = []
                pose_det, pairs = get_pose_from_json(pose_det, 0.0, 0)
                if len(pose_det) > 0:
                    for j in range(len(det_list_all[i])):
                        x1, y1, x2, y2 = det_list_all[i][j].curr_rect.astype(int)
                        trks.append([x1, y1, x2, y2])
                    kp = pose_det
                    match = assign_keypoint(kp, trks)

                    for m, n in match:
                        kp_id = kp[m, 0]
                        det_list_all[i][n].conf = 1
                        if kp_id not in pose_tracklets:
                            pose_tracklets[kp_id] = pose_tracklet(kp_id,kp[m, 0],det_list_all[i][n])
                        else:
                            pose_tracklets[kp_id].add_trk(kp[m, 1:], kp[m, 0], det_list_all[i][n])
                        cur_pose_tracklet[n] = pose_tracklets[kp_id]

                    if len(match) > 0 and len(pairs) > 0:
                        for pair in pairs:
                            if pair[0] in match[:,0] and pair[1] in match[:,0]:
                                pair_hand_1 = match[match[:,0]==pair[0]][0][1]
                                pair_hand_2 = match[match[:, 0] == pair[1]][0][1]
                                pose_tracklets[kp[pair[0],0]].add_pair(det_list_all[i][pair_hand_1])
                                pose_tracklets[kp[pair[1], 0]].add_pair(det_list_all[i][pair_hand_2])
                                cur_hands_paris[pair_hand_2] = [pair_hand_1, pose_tracklets[kp[pair[0], 0]]]
                                cur_hands_paris[pair_hand_1] = [pair_hand_2, pose_tracklets[kp[pair[1], 0]]]
            else:
                pose_det = []
            ############################################################################################

            if i == 0:
                for j in range(len(det_list_all[i])):
                    if det_list_all[i][j].conf > init_threshold:
                        det_list_all[i][j].id = j + 1
                        max_id = max(max_id, j + 1)
                        track = tracklet(det_list_all[i][j])
                        tracklet_all.append(track)
                continue

            matches, unmatched1, unmatched2 = track_det_match(tracklet_all, det_list_all[i], IOU_threshold, l2=False, prob=True)

            # second_matches
            remaining_tracklet = [tracklet_all[k] for k in unmatched1]
            second_matches, _, _ = track_det_match(remaining_tracklet, low_score_det, rematch_threshold, l2=False, prob=True)
            for j in range(len(second_matches)):
                match1, match2 = second_matches[j]     
                if len(matches) == 0:
                    matches = np.array([[unmatched1[match1], len(det_list_all[i])]])
                else:
                    matches = np.concatenate((matches, np.array([[unmatched1[match1], len(det_list_all[i])]])), 0)
                det_list_all[i].append(low_score_det[match2]) 

            if len(matches)>0 and len(np.unique(matches[:,1])) != len(matches):
                pdb.set_trace()
            if len(second_matches) > 0:
                unmatched1 = np.delete(unmatched1, second_matches[:, 0])
            
            matched_id_list = []
            for j in range(len(matches)):
                last_frame = tracklet_all[matches[j][0]].last_frame
                if last_frame < i-1 and i-last_frame < interploate_gap:
                    last_bbox = tracklet_all[matches[j][0]].last_rect.curr_rect
                    cur_bbox = det_list_all[i][matches[j][1]].curr_rect
                    for k in range(last_frame+1, i):
                        k_box = last_bbox + (cur_bbox-last_bbox)/(i-last_frame-1)*(k-last_frame)
                        k_box = processed_bbox(k_box, w, h)
                        det_rect = set_det_rect(k_box, 1, np.zeros([4]), k)
                        det_rect.id = tracklet_all[matches[j][0]].id
                        det_list_all[k].append(det_rect)
                        tracklet_all[matches[j][0]].add_rect(det_rect)

                det_list_all[i][matches[j][1]].id = tracklet_all[matches[j][0]].id
                tracklet_all[matches[j][0]].add_rect(det_list_all[i][matches[j][1]])
                matched_id_list.append(det_list_all[i][matches[j][1]].id)
                continue

            for j in range(len(unmatched2)-1, -1, -1):
                if pose_dets['score'] > 1.1:
                    if unmatched2[j] in cur_pose_tracklet and cur_pose_tracklet[unmatched2[j]].get_recent_hand_id not in matched_id_list:
                        if cur_pose_tracklet[unmatched2[j]].get_recent_hand_id == 0:
                            continue
                        det_list_all[i][unmatched2[j]].id = cur_pose_tracklet[unmatched2[j]].get_recent_hand_id
                        matched_id_list.append(det_list_all[i][unmatched2[j]].id)
                        for k in range(len(unmatched1)-1, -1, -1):
                            if tracklet_all[unmatched1[k]].id == det_list_all[i][unmatched2[j]].id:
                                tracklet_all[unmatched1[k]].add_rect(det_list_all[i][unmatched2[j]])
                                unmatched1 = np.delete(unmatched1, k)
                        unmatched2 = np.delete(unmatched2, j)

            for j in range(len(unmatched2)):
                if unmatched2[j] in cur_hands_paris and cur_hands_paris[unmatched2[j]][0] not in cur_hands_paris and pose_dets['score'] > 1.1 and cur_pose_tracklet[unmatched2[j]].get_pair_hand_id not in matched_id_list: 
                    det_list_all[i][unmatched2[j]].id = cur_hands_paris[unmatched2[j]][1].get_pair_hand_id
                    matched_id_list.append(det_list_all[i][unmatched2[j]].id)
                    for k in range(len(unmatched1) - 1, -1, -1):
                        if tracklet_all[unmatched1[k]].id == det_list_all[i][unmatched2[j]].id:
                            tracklet_all[unmatched1[k]].add_rect(det_list_all[i][unmatched2[j]])
                            unmatched1 = np.delete(unmatched1, k)
                    continue

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
    # **************save evaluate file**************  **
    fout_tracking = open(os.path.join(out_dir, vid_name + '.txt'), 'w')
    for i in range(img_len):
        ids = []
        for j in range(len(det_list_all[i])):
            x1, y1, x2, y2 = det_list_all[i][j].curr_rect.astype(int)
            trace_id = det_list_all[i][j].id   
            if trace_id == 0:
                continue
            ids.append(trace_id)
            fout_tracking.write(
                str(i+1) + ',' + str(trace_id) + ',' + str(x1) + ',' + str(y1) + ',' + str(x2 - x1) + ',' + str(
                    y2 - y1) + ',-1,-1,-1,-1\n')

    fout_tracking.close()
    print('Speed in HZ {}'.format((img_len+1)/(time.time() - start_time)))

def main():
    args = parse_args()
    cfg = get_cfg()
    add_handler_config(cfg)
    cfg.merge_from_file(args.cfg)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model 
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    print("load weights from " + cfg.MODEL.WEIGHTS)
    videos_path = args.root_path
    split = 'test'

    dirs = os.listdir(videos_path+split)
    dirs.sort(key=str.lower)

    model = CustomPredictor(cfg, True)
    for idx, seq_num in enumerate(dirs):
        run_each_dataset(args, model, videos_path, split, seq_num, args.out_dir)

if __name__ == '__main__':
    main()