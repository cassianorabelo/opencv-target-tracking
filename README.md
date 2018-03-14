# opencv-target-tracking

![Computer vision RC plane tracking](https://i.imgur.com/MnvUcdB.gif)

The following code was developed as an exercise to learn how to use the OpenCV Computer Vision Library in C++. The proposed objective was to detect and track, in realtime a given 2D pattern (target). There can never be a false positive.

![2D target to be tracked](https://i.imgur.com/TYxTJWx.png?1)

Due to the complexity of the job at hand, it was necessary to divide it in a series of steps that in the end converge to a satisfactory solution. The following figure shows some of the adopted steps.

![Some of the steps taken to track the target.](https://i.imgur.com/aHiTgQh.jpg)

- (a) Raw footage showing the target that must be tracked and a QR code that cannot be detected (as everything else in the scene except the target);
- (b) conversion from color to grayscale;
- (c) binarization;
- (d) contour detection;
- (e) Filtering based on diverse geometrical properties;
- (f) Perspective correction;
- (g) Filtering based on circle detection;
- (h) Filtering based on the existence of straight lines at certain angles;
- (i) Displaying the detected target.

In order to improve performance, it was decided to parallelize certain parts of the code. Since [findContours](https://docs.opencv.org/3.1.0/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a) operation generates a large number of contours and we only want those with specific characteristics, each contour can be evaluated independently of the rest, which is an ideal situation for the use of parallel processing. OpenCV has handy class for parallel data processors called [ParallelLoopBody](https://docs.opencv.org/3.4.0/d2/d74/classcv_1_1ParallelLoopBody.html#details). For this part of the code to work, it is necessary to compile OpenCV with support for the TBB - Threading Building Blocks library from Intel.
Parts of the code were based on the [ArUco module](https://docs.opencv.org/3.1.0/d9/d53/aruco_8hpp.html) from OpenCV.

![Detection result with debug mode on](https://i.imgur.com/3y6uCnj.jpg)
