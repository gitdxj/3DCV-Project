# 3DCV-Project

This is a re-implementation of the ["Robust 6D Object Pose Estimation by 
Learning RGB-D Features"](https://arxiv.org/abs/2003.00188) paper. The original implementation can be found on
[https://github.com/mentian/object-posenet](https://github.com/mentian/object-posenet).

## Data
The pose estimation network is trained on the LINEMOD dataset. A preprocessed version of
this dataset, that also contains object segmentations obtained by a SegNet, is available
[here](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7).

## Trained Model
Our trained model can be downloaded [here](https://drive.google.com/file/d/1nzxOrGweL-mALBnItxPHB0CIkfMZw212/view?usp=sharing).

## How to run the demo notebook
Install the requirements from requirements.txt, download the dataset and the trained model
and adjust the paths to the dataset directory and to the trained model if necessary.

## Training
Cuda is required for training. Adjust the dataset_path in train_linemod in train.py and then 
run train.py

## Evaluation
Cuda is required for evaluation. Adjust the dataset_path and path_to_trained_model in 
evaluate.py and then run evaluate.py