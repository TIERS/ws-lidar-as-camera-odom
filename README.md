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

## install cpu monitor
```
git clone https://github.com/alspitz/cpu_monitor.git
catkin build
source devel/setup.bash
```

## Optimize step 
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
Run the pointcloud matching approach for the odometry 
```
roslaunch kiss_icp odometry.launch topic:=/keypoint_point_cloud
```

Play the rosbag

```
rosbag play [rosbag that you have]
```

Run CPU_monitor

```
roslaunch cpu_monitor cpu_monitor.launch poll_period:=1
```

Run python scripts to get the npy file

```
python3 cpu_mem.py
```

Run python scripts to get the results

```
python3 plot_cpu_mem.py
```

### KISS-ICP with raw point cloud

![](./imgs/kiss-icp-raw.png)

### KISS-ICP with conventional keypoint extractors

![](./imgs/kiss-icp-a.png)

### KISS-ICP with the superpoint feature extractor

