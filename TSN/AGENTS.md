# Implement a Temporal Segment Networks (TSN)
There are 3 sub-tasks in this task, each with different points given if complete, prioritize higher points.

## RGB TSN (5 points)
Implement a TSN network only for RGB signals with a [ResNet-18 backbone](https://docs.pytorch.org/vision/stable/models.html#torchvision.models.resnet18),  which is initialized from ImageNet (you may use official PyTorch implementation). Use the following hyperparameters:
1. Number of segments = 4.
2. During training randomly select the snippet inside each segment.
3. During testing select the middle snippet inside each segment.
4. Use the same input size and input preprocessing for the RGB frames.

Deductions:
- -1 point – didn’t use pretrained resnet18 
- -1 point – frame sampling during training/testing is incorrect 
- -0.5 point – mistakes, e.g, in sampling of segments during training 
- -2 point – only code if provided, no training results

## Optical Flow TSN (5 points)
Implement a TSN network only for optical flow. Try 2 settings:
- initialization  from ImageNet (how should one initialize the first layer?)-
- random initialization.

Compare the training and validation loss figures of the 2 settings.  (Between the 2 settings, only change the kind of initialization, keep all other hyperparameters the same). Use the following hyperparameters:
1. Number of segments = 4.
2. Use a stack of 7 consecutive x,y flows as input. This results in a 14 channel input (as compared to the RGB settings that you have 3 channel input)

Deductions:
- -1 point – only random initialization setting was tested
- -0.5 point – in ImageNet initialization setup, the first layer is not initialized

## RGB vs. Optical Flow and Fusion (2.5 points)
Compare the performance of RGB and Optical Flow models for each class. Use late fusion (averaging the probability estimate of RGB and optical flow models at test time) to obtain a single prediction from the 2 models trained on the 2 different modalities. Can we achieve higher performance?

Deductions:
- -1 point – per-class performance is not compared
- -1 point - not late fusion results