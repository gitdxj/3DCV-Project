# 3DCV-Project

This is a re-implementation of the ["Robust 6D Object Pose Estimation by 
Learning RGB-D Features"](https://arxiv.org/abs/2003.00188) paper. The original implementation can be found on
[https://github.com/mentian/object-posenet](https://github.com/mentian/object-posenet).  

The PSPNet under model/psp is from https://github.com/Lextal/pspnet-pytorch.

## Data
The pose estimation network is trained on the LINEMOD dataset. A preprocessed version of
this dataset, that also contains object segmentations obtained by a SegNet, is available
[here](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7). 
This preprocessed dataset is used during evaluation.

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

### A note on evaluation
We have encountered an issue with calling ``model.eval()`` after loading the 
trained model. When doing this, the accuracy of the predictions is nearly 0. It seems
that this is a known issue ([https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323](https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323))
and might have to do with BatchNorm layers. However, we did not have enough time to 
investigate this further, which is why in the demo notebook and in evaluate.py we did
not call ``model.eval()``.