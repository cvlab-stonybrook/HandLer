import os
import shutil
from_path = '/archive/mingzhen/dataset/hand_det/ContactHands/Annotations/'
to_path = '/archive/mingzhen/dataset/hand_det/voc_anno'

for file in os.listdir(from_path):
    to_file = os.path.join(to_path, file)
    shutil.move(os.path.join(from_path, file), to_file)
