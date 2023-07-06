# signal-image-keypoints-odom

## Installation

### Clone our repo
```
git clone --recurse-submodules git@github.com:RealYXJ/ws-lidar-as-camera-odom.git
cd ws-lidar-as-camera-odom
catkin build 
```

### Install Dependency
```
<!-- libtorch -->
cd ws-lidar-as-camera-odom/src/ && mkdir libs && cd libs
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip
rm libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip
```
Note: The lastest version will cause problems like unference to ros::init even though I have linked with `${catkin_LIBRARIES}`. Same problem has met by other people in https://github.com/pytorch/pytorch/issues/60178. I followed the solution provided by mmurooka commented on Nov 2, 2022 (https://github.com/mmurooka/SMPLpp/issues/1). Issue solved.

## Run 
```
source ws-lidar-as-camera-odom/devel/setup.bash
```

Run the ROS master in one terminal 
```
roscore
```

Run the keypoint extractor ros node by either
```
roslaunch signal-image-keypoints-odom signal-feature-odom.launch
```
or
```
cd ws-lidar-as-camera-odom/src/signal-image-keypoints-odom/superpoint_extractors
# in the proper python envrionment
python superpoint_evaluation.py
```

Run the pointcloud matching approach for the odometry 
```
roslaunch kiss_icp odometry.launch topic:=/keypoint_point_cloud
```

(Optional) if you would like to run kiss-icp directly with raw point cloud.
```
roslaunch kiss_icp odometry.launch topic:=/os_cloud_node/points

```

Play the rosbag
```
rosbag play indoor01_square.bag
```

## Results

### KISS-ICP with raw point cloud

![](./imgs/kiss-icp-raw.png)

### KISS-ICP with conventional keypoint extractors

![](./imgs/kiss-icp-a.png)

### KISS-ICP with the superpoint feature extractor


