# signal-image-keypoints-odom

## Installation

### Clone our repo
```
git clone --recurse-submodules git@github.com:RealYXJ/ws-lidar-as-camera-odom.git
cd ws-lidar-as-camera-odom
git checkout optimize_code_zhz
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
cd ws-lidar-as-camera-odom/src/
git clone https://github.com/alspitz/cpu_monitor.git
```

## cmake build oand run our project
```
cd ws-lidar-as-camera-odom/
catkin build
```


## Optimize step 
```
cd ws-lidar-as-camera-odom/
source devel/setup.bash
```

Run the ROS master in one terminal 
```
roscore
```

Run the keypoint extractor ros node
```
roslaunch signal-image-keypoints-odom signal-feature-odom.launch
```
Run the pointcloud matching approach for the odometry, or you can choose topic:=//os_cloud_node/points

```
roslaunch kiss_icp odometry.launch topic:=/keypoint_point_cloud
```

Run CPU_monitor

```
roslaunch cpu_monitor cpu_monitor.launch poll_period:=1
```

Run python scripts to show CPU and memory usage of our nodes, cpu_mem2.py will print the current mean value of our nodes.

```
python3 cpu_mem2.py
```

Play the rosbag

```
rosbag play [rosbag that you have]
```



### KISS-ICP with raw point cloud

![](./imgs/kiss-icp-raw.png)

### KISS-ICP with conventional keypoint extractors

![](./imgs/kiss-icp-a.png)

### KISS-ICP with the superpoint feature extractor

