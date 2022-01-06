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

If you are windows user, you can still run the pipeline for all three dataset. But in order to do so, you need to open src/main/main.py file and uncomment the config of dersired dataset.

To run on parking, just uncomment the parking config path.
```
CONFIG_PATH = "src/main/configs/Parking.yaml"
#CONFIG_PATH = "src/main/configs/malaga.yaml"
#CONFIG_PATH = "src/main/configs/KITTI.yaml" 
```
To run on parking, just uncomment the malaga config path.
```
#CONFIG_PATH = "src/main/configs/Parking.yaml"
CONFIG_PATH = "src/main/configs/malaga.yaml"
#CONFIG_PATH = "src/main/configs/KITTI.yaml" 
```
To run on parking, just uncomment the kitti config path.
```
#CONFIG_PATH = "src/main/configs/Parking.yaml"
#CONFIG_PATH = "src/main/configs/malaga.yaml"
CONFIG_PATH = "src/main/configs/KITTI.yaml" 
```