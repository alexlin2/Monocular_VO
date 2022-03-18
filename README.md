# Monocular_VO
## Abstract

Determining the trajectory or direction of travel of a mobile robot or car is essential for navigation and and movement. If you have ever driven a car then you would notice a number on your dashboard that tells you how far your car has travel in it's lifespan. This is called an odometer. This information is essential for mechanics to diagnose the health and deterioration of your vehicle and it determines your car's used value on the second hand market. Odometry is also vital for robot navigation, this information is used to determine the state of the robot at any given time. However, odometry is often hard to do without supporting hardware, and inaccurate when using a traditional wheel speed sensor. This project aims to do simple odometry using just a single rgb camera. It solves a complex problem using just software and knowledge of computer vision. 

# Problem statement

Odometry in essense is determining the distance traveled and the direction of travel of the camera at any given time. So for every time instance $t$, we want to determine the pose vector $[x^{t} y^{t} z^{t} \alpha^{t} \beta^{t} \gamma^{t}]$ that describes the position and orientation of the camera. Note that $\alpha^{t}, \beta^{t}, \gamma^{t}$ is in Euler angles, and $x^{t}, y^{t}, z^{t} $ is in Cartesian coordinates. For this problem, we are given the initial known position, and the initial orientation of the camera. They are given as a 3 by 3 rotation matrix $R_{pos}$ and a 3 by 1 translation vector $t_{pos}$. Camera instrinsic such as focal length and the principle point will also be known, and we are assuming a pinhole camera model. 

# Algorithm

We will use Nister's Five Point Algorithm for Essential Matrix estimation. The essential matrix is a 3x3 matrix that relates corresponding points in two images, it is similar to the homography matrix that we learned in class. We will also use the FAST(Features from accelerated segment test) algorithm to detect corners and features in the image, it is similar to the ORB features that we learned in class. 

Here is an outline of the algorithm:

1. Capture two consecutive images at $t$ and $t+1$
2. Use `cv2.FastFeatureDetector` to detect keypoints in the first image at time $t$
3. Use LK optical flow `cv2.calcOpticalFlowPyrLK` to determine the corresponding keypoints in the second image at time $t+1$ 
4. Use Nister's Five Point Algorithm `cv2.findEssentialMat` with RANSAC to find the essential matrix that describe the corresponding point from image $t$ to image $t+1$
5. Estimate the rotation matrix $R$ and the translation vector $t$ from the essential matrix that was obtained in the last step.
6. Use the formula $R_{pos}^{t+1} = RR_{pos}^{t}$ and $t_{pos}^{t+1} = t_{pos}^{t} + tR_{pos}$ to determine the new pose vector that describes the position and orientation of the camera

# Dataset

The dataset used is [KITTI Visual Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) 