Pollen counting model

This demo shows bee and pollen sacs detection, draw the bee flying trajectory and pollen count number.

Mention: the model running very slow. Please wait 5s, the image will be come out. The main problem is that the Hough transform coding is not vectorized. The "for loop" slow down the programme.

The Matlab essential tools: 
   Image processing tool box
   computer vision tool box
   deep learning tool box
   neural network tool box
The model has been tested in Matlab 2018b.

Methodology
 Motion detection and colour thresholding to detect bees
 Kalman filter and Hungarian method to track bees.
 Hough transform to find bee position in merge bee blobs.
 Faster RCNN model to detect pollen sacs


Windows
 Yellow bounding boxes: single bee detection.
 Red bounding boxes: Bee position prediction for the current frame
 Cyan boudning boxes: Kalman correction of bee location
 Purple small bounding boxes: pollen sacs detection with probalility
 Purple circle with numbers on the left bottom of bounding box:
 Number followiing format pol/sig:
    pol: pollen detection number in frames
    sig: number of bee detected as single bee in frames
    Problablity of bee carrying pollen is pol/sig*100 (%)
    If the % greater than 46%, the bee would be identified carrying pollen
    sacs. The threshold of 46% was got from ROC analysis. More detrail see
    the thesis.
