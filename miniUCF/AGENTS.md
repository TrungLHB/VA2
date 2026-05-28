## miniUCF
Implementation of the PyTorch Dataset that you will use to load the data. This could potentially be 2 classes, 1 for RGB frames and 1 for optical flow or 1 class that takes the modality as an initializer argument.

The dataset is provided as 2 zip files. One containing the original AVI videos which you can use to extract the RGB frames. And another one containing pre-computed optical flows. Alongside 2 zip files, 3 .txt files are also provided containing the following information:
1. classes.txt: Contains the class id, class name mapping.
2. train.txt: Contains the list of videos that should be used for training.
3. validation.txt: Contains the list of videos that should be used for evaluation.

Each line in train.txt or validation.txt is a video identifier, something like: CLASS_NAME/V IDEO_NAME
where CLASS_NAME is the name of the class that the video belongs to, and you can find the mapping to class id in classes.txt (e.g. ApplyEye- Makeup). And where VIDEO_NAME is the name of the video (e.g.  v_ApplyEyeMakeup_g08_c01).

Using the video identifier one can access the video AVI file by accessing: CLASS_NAME/V IDEO_NAME.avi.

The flow files for each video identifier can be accessed from:
- CLASS_NAME/V IDEO_NAME/flow_x_0001.jpg
- CLASS_NAME/V IDEO_NAME/flow_x_0002.jpg
- ...
- CLASS_NAME/V IDEO_NAME/flow_x_N.jpg

where N is the maximum number of frames in each video. Notice that the flow in each direction is saved as a compressed single-channel .jpg file which you can use OpenCV to load.

## Important
Read ahead to Task 1 and 2 to see if there is a need to split the custom dataset implementation into two different dataset.py file for each of those task