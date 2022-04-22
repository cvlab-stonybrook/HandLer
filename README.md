
# HandLer

This repository contains the code and data for the following paper:

[Forward Propagation, Backward Regression, and Pose Association for Hand Tracking in the Wild
](https://www3.cs.stonybrook.edu/~hling/publication/hand-cvpr22.pdf) (CVPR 2022).

<p align="center">
    <figure>
        <img src="teaser.gif" height="310" width="480" />
            <figcaption> We develop a method to detect and track multiple hands in videos. </figcaption>
     </figure>
</p>

## Installation

Install Torch 1.7.1
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Install other requirements 
```
pip install -r requirements.txt
```

Install Detectron2 (verizon 0.4)

For CUDA 11.0 and Torch 1.7.x
```
python -m pip install detectron2==0.4 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
```

## Youtube-Hand Dataset

Please find the Youtube-Hand dataset in the [project page](https://mingzhenhuang.com/projects/handler.html).


## Go to main folder
```
cd ./projects/HandLer
```

## Model Weights

Please find the pretraind model weights here:

```
https://drive.google.com/file/d/1eaZn9E7lHvXY3Fh90f2eIK1P2yZwNfnP/view?usp=sharing
```

Please find the weights trained on Youtube-Hand training set here:

```
https://drive.google.com/file/d/1sK5bRTHt5zLOVPruUt-zyZ_AOrBvGQ9-/view?usp=sharing
```

## Hand Tracking Demo: Run HandLer on custom videos
```
python demo.py --cfg configs/yt_trk.yaml --output output_vid.mp4 --weights /path/to/weights --input input_vid.mp4
```

## Training on Youtube-Hand Dataset

For training on single GPU:

```
python train_trk.py --config-file configs/yt_trk.yaml
```

For training on multiple GPUs:

```
python train_trk.py --config-file configs/yt_trk.yaml --num_gpu 4
```

## Testing on Youtube-Hand Test set

```
python tracking.py --cfg /path/to/cfg --out_dir /path/to/results/folders --weights /path/to/weights --root_path /path/to/dataset
```

## Evaluating on Youtube-Hand Test set

```
python evaluations.py /path/to/results/folders
```

## Improving Hand Tracking using Hand-Body Association

By associating hands with human bodies and linking human bodies across frames, we can establish correspondence between detected instances of the same hand across different frames, reducing identity switches in tracking. To know more about this, please check our work [Whose Hands Are These? Hand Detection and Hand-Body Association in the Wild](http://vision.cs.stonybrook.edu/~supreeth/BodyHands/)


## Citation

If you find our code or dataset useful, please cite:

```
@inproceedings{handler_2022,
      title={Forward Propagation, Backward Regression and Pose Association for Hand Tracking in the Wild},
      author={Mingzhen Huang and Supreeth Narasimhaswamy and Saif Vazir and Haibin Ling and Minh Hoai},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2022},
  }

@inproceedings{bodyhands_2022,
      title={Whose Hands Are These? Hand Detection and Hand-Body Association in the Wild},
      author={Supreeth Narasimhaswamy and Thanh Nguyen and Mingzhen Huang and Minh Hoai},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2022},
  }
```
