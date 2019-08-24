Pollen counting model

This demo shows bee and pollen sacs detection, draw the bee flying trajectory and pollen count number.

Mention: the model running very slow. Please wait 5s, the image will be come out. The main problem is that the Hough transform coding is not vectorized. The "for loop" slow down the programme.

The Matlab essential tools:
  Image processing tool box,
  computer vision tool box,
  deep learning tool box,
  neural network tool box,
The model has been tested in Matlab 2018b.

Methodology:
 Motion detection and colour thresholding to detect bees.
 Kalman filter and Hungarian method to track bees.
 Hough transform to find bee position in merge bee blobs.
 Faster RCNN model to detect pollen sacs.


Windows:
 Yellow bounding boxes: single bee detection.
 Red bounding boxes: Bee position prediction for the current frame.
 Cyan boudning boxes: Kalman correction of bee location.
 Purple small bounding boxes: pollen sacs detection with probalility.
 Purple circle with numbers on the left bottom of bounding box.
 Number followiing format pol/sig:
    pol: pollen detection number in frames;
    sig: number of bee detected as single bee in frames;
    Problablity of bee carrying pollen is pol/sig*100 (%);
    If the % greater than 46%, the bee would be identified carrying pollen
    sacs, The threshold of 46% was got from ROC analysis. More detrail see
    the thesis.

Please download FaterRCNN mode in this link: https://drive.google.com/open?id=1G-E1128S9tJrxUbWfwYe59XzIXl29AJd

Please download test videos from the list below:
Test video 1: https://drive.google.com/open?id=1QdmWUeh4OYRivpZjn6749H8ovikrfF3I

Test video 2: https://drive.google.com/open?id=1nI9vdQ8tz7MnjFawcuh-_1m04Wk6GZmY

Test video 3: https://drive.google.com/open?id=1gmW569-YoqRHD5ePqa7849D5CWKv1uo0

Test video 4: https://drive.google.com/open?id=1khfXQ_NndokejMOIXoY1vXeC-CWbQVij

Test video 5: https://drive.google.com/open?id=1fdQUxreJ31liBKIxsLNaJA86iu19-ZIN
