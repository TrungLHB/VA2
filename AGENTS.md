## Task
In this exercise, we will focus on using Convolutional Neural Networks (CNNs) for action recognition, i.e. classification of the action being performed in a short video. The dataset used is a subset of the UCF101 dataset with 25 classes which we call miniUCF. The train-validation split is already provided as well as precomputed optical flows. RGB frames have to be extracted from the AVI files.

You should implement 2 different models for action recognition: TSN and 3D-ResNet. The input to each can be RGB frames or optical flow. Each network can also be randomly initialized or initialized from ImageNet.

## Requirements
- Use Python3.
- You should use PyTorch >= 1.1 and torchvision.

## Remarks
Please consider writing clear and well-documented code. The easier to read and understand your code is, the lower the probability of incorrectly getting a lower grade than you deserve.

This is meant to be a student project so code readability is much more important than re-usability, especially if the use-cases go beyond the required tasks. 

Complete the project in parts.

1. Task 0: implement a custom dataset
2. Task 1: implement a Temporal Segment Networks (implement in TSN/)
3. Task 2: implement an RGB 3D ResNet (implement in 3DResNet/)

You may read ahead make sure everything is still compatible between tasks but only move on to the next task when I say so.