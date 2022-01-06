# vamr-mini-project
This repository contains VAMR mini project problem statement, dataset, proposed solution, solution report.

## Details
This repository implements a simple, monocular, visual odometry (VO) pipeline with the most essential features: initialization of 3D landmarks, keypoint tracking between two frames, pose estimation using established 2D ↔ 3D correspondences, and triangulation of new landmarks.

You can run the repo for three datasets available in a folder named data.

If you are linux or macOs user, you can directly run the pipeline for three data using bash files. Just do the following:

To run on malaga dataset, do the following:
```
chmod +x malaga.sh
./malaga.sh
```
To run on parking dataset, do the following:
```
chmod +x parking.sh
./parking.sh
```
To run on KITTI dataset, do the following:
```
chmod +x kitti.sh
./kitti.sh
```
In case you wanna run for a dataset multiple times, you do not to run chmod command multiple times. chmod command only make a file executable. 

In case you are windows user. 