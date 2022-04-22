import os
import json
import numpy as np
import copy, pdb
import motmetrics as mm
mm.lap.default_solver = 'lap'

from tracking_utils.io import read_results, unzip_objs


class Evaluator(object):

    def __init__(self, data_root, seq_name, data_type, pose=False, bsl=False):
        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type
        self.is_pose = pose
        self.is_bsl = bsl

        self.load_annotations()
        self.reset_accumulator()

    def load_annotations(self):
        assert self.data_type == 'mot'
        self.anno_frame = None
        if self.is_pose:
            self.anno_frame = []
            pose_gt_file = os.path.join('/archive/mingzhen/dataset/Youtube-Hands/posetrack_annotation', self.seq_name+'.json')
            with open(pose_gt_file, 'r') as f:
                pose_gt = json.load(f)
            for idx, img in enumerate(pose_gt['images']):
                if img['is_labeled']:
                    self.anno_frame.append(idx+1)
        elif self.is_bsl:
            gt_file = os.path.join('/archive/mingzhen/dataset/bsl/images/test/sample/gt/gt.txt')
            gt = np.loadtxt(gt_file, delimiter=',')
            self.anno_frame = list(gt[:,0])
            self.anno_frame = [int(i) for i in self.anno_frame]
            # import pdb; pdb.set_trace()

        gt_filename = os.path.join(self.data_root, self.seq_name, 'gt', 'gt.txt')
        self.gt_frame_dict = read_results(gt_filename, self.data_type, is_gt=True, anno_frame=self.anno_frame)
        # import pdb; pdb.set_trace()
        self.gt_ignore_frame_dict = read_results(gt_filename, self.data_type, is_ignore=True)



    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def l2_dist_to_iou(self, gt_tlwhs, trk_tlwhs, dist):
        gt_centers = gt_tlwhs[:, :2] + gt_tlwhs[:,2:]/2
        trk_centers = trk_tlwhs[:, :2] + trk_tlwhs[:,2:]/2
        dists = np.zeros((gt_centers.shape[0], trk_centers.shape[0]))
        for i in range(gt_centers.shape[0]):
            # pdb.set_tracqe()
            dist = 1.2*np.linalg.norm(gt_tlwhs[i][2:])
            distance=np.sqrt(np.sum(np.square(trk_centers - gt_centers[i,:]),axis=1))
            distance[distance>dist] = np.nan
            dists[i,:]=1-distance/dist
        # pdb.set_trace()
        return dists

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

        # ignore boxes
        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = unzip_objs(ignore_objs)[0]

        # remove ignored results
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        iou_distance = mm.distances.iou_matrix(ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if len(iou_distance) > 0:
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
            match_ious = iou_distance[match_is, match_js]

            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]
            keep[match_js] = False
            trk_tlwhs = trk_tlwhs[keep]
            trk_ids = trk_ids[keep]
        #match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
        #match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
        #match_ious = iou_distance[match_is, match_js]

        #match_js = np.asarray(match_js, dtype=int)
        #match_js = match_js[np.logical_not(np.isnan(match_ious))]
        #keep[match_js] = False
        #trk_tlwhs = trk_tlwhs[keep]
        #trk_ids = trk_ids[keep]

        # get distance matrix
        # iou_distance = self.l2_dist_to_iou(gt_tlwhs, trk_tlwhs, dist = 200)
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, 'last_mot_events'):
            events = self.acc.last_mot_events  # only supported by https://github.com/longcw/py-motmetrics
        else:
            events = None
        return events

   

    def eval_file(self, filename):
        self.reset_accumulator()

        result_frame_dict = read_results(filename, self.data_type, is_gt=False, anno_frame=self.anno_frame)
        # import pdb; pdb.set_trace()
        if len(result_frame_dict) == 0:
            result_frame_dict = {1: [((602.0, 266.0, 40.0, 86.0), 0, 1)]}

        frames = sorted(list(set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys())))
        empty = True
        for frame_id in frames:
            if frame_id % 15 != 1:
                if "viva" not in filename and 'mpii' not in filename and not self.is_bsl:
                    continue 
                # elif frame_id % 3 != 1:
                #     continue
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            self.eval_frame(frame_id, trk_tlwhs, trk_ids, rtn_events=False)

        return self.acc

    @staticmethod
    def get_summary(accs, names, metrics=('mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )

        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()
