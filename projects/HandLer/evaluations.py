import motmetrics as mm
import numpy as np
import pdb
import pandas as pd
import argparse
import os
import sys, pdb
from pathlib import Path
from tracking_utils.evaluation import Evaluator
accs=[]
path = '/nfs/bigneuron/mingzhen/dataset/Youtube-Hands/images/test'
res_path = sys.argv[1] + '/'
dirs = os.listdir(path)
dirs = [i.split('.')[0] for i in dirs]
dirs.sort(key=str.lower)
for dir in reversed(dirs):
    filePath = res_path + dir + '.txt'
    evaluator = Evaluator(path, dir, 'mot')
    accs.append(evaluator.eval_file(filePath))

metrics = mm.metrics.motchallenge_metrics
mh = mm.metrics.create()
dirs = list(reversed(dirs))
summary = Evaluator.get_summary(accs, dirs ,metrics)
strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)