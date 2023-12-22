# Signal-image-keypoints-odom


![Alt text](/imgs/Kiss_55_sup.png)

## Introduction
We present a novel approach to estimate the LiDAR-based odometry by using signal image keypoints. 

Our work contributes in three main aspects:

* We conduct an extensive study of the optimal resolution and interpolation approaches for enhancing the low-resolution LiDAR-generated data to extract keypoints more effectively.

* We investigate the efficacy of the existing keypoint detectors and descriptors on LiDAR-generated images with multiple specialized metrics providing a quantitative evaluation.
  
  For more details on these two aspects, please refer to this [folder](detector_descriptor_for_LiDARimg), which contains detailed introductions and experimental results.

* We propose a novel approach that leverages the detected keypoints and their neighbors to extract a reliable point cloud (downsampling) for the purpose of point cloud registration with reduced computational overhead and fewer deficiencies in valuable point acquisition.

For more details, please refer to our paper: [LiDAR-Generated Images Derived Keypoints Assisted Point Cloud Registration Scheme in Odometry Estimation](https://www.mdpi.com/2072-4292/15/20/5074).


## Repository Structure
- `detector_descriptor_for_LiDARimg`: The first two contributions of our work, firstly, we explore a preprocessing method for enhancing low-resolution LiDAR-generated data to extract keypoints more effectively. Second and more important, we assess the effectiveness of the current keypoint detectors and descriptors on LiDAR-generated images using multiple specialized metrics. Please refer to the more detailed [README](detector_descriptor_for_LiDARimg/README.md) inside the folder.
- `src`: 
    * signal-image-keypoints-odom: Source code of our new odometry approach.
    * libs: libtorch--The C++ version for Pytorch.
    * kiss-icp: The original kiss-icp (After you run 'git submodule update --init --recursive')
    * cpu_monitor: A cpu monitor for monitoring the performance of our method. (After you run 'git submodule update --init --recursive')


- `scripts`: several python scripts to print out the CPU and memory usage of our method. you don't have to run them.


And the original rosbag dataset we used is available at the [University of Turku servers](https://utufi.sharepoint.com/:f:/s/msteams_0ed7e9/Etwsa7m8hxhMk9H3x-K6DfUBgU3x-ZK9vMeD_V0J2mdHwA).




## Run our code

### Clone our repo
```
git clone git@github.com:RealYXJ/ws-lidar-as-camera-odom.git
cd ws-lidar-as-camera-odom
```
### Update the submodules

Since we have used some submodules, so we need to update the submodules.
```
git submodule update --init --recursive
```

### Install libtorch

```
cd ws-lidar-as-camera-odom/src/ && mkdir libs && cd libs

wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip

unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip

rm libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip
```

Note: The lastest version will cause problems like unference to ros::init even though I have linked with `${catkin_LIBRARIES}`. Same problem has met by other people in https://github.com/pytorch/pytorch/issues/60178. I followed the solution provided by mmurooka commented on Nov 2, 2022 (https://github.com/mmurooka/SMPLpp/issues/1). Issue solved.


### Other requirements

The code has been tested on Ubuntu 20.04 with ROS Noetic. And we use the following common libraries. You can check the [CMakeLists.txt](src/signal-image-keypoints-odom/CMakeLists.txt) for more details.

* opencv
* pcl


### Make sure you have the right path


We found that there is a path in our code, it has to be the absolute path, so you need to change it to your own path.
This line of code is inside the [signal_image_keypoints_odom.hpp](src/signal-image-keypoints-odom/include/signal_image_keypoints_odom/signal_image_keypoints_odom.hpp) file.

Line 79:
```cpp
  std::string weight_path_ = "/home/jimmy/Downloads/ws-lidar-as-camera-odom/src/signal-image-keypoints-odom/model/superpoint_v2.pt";
```


### cmake build
```
cd ws-lidar-as-camera-odom/
catkin build
source devel/setup.bash
```


### Run the keypoint extractor ros node
```
roslaunch signal-image-keypoints-odom signal-feature-odom.launch
```


### Run the pointcloud matching approach 

Run the pointcloud matching approach for the odometry ( or you can choose topic:=//os_cloud_node/points, to try the kiss-icp with raw point cloud)

```
roslaunch kiss_icp odometry.launch topic:=/keypoint_point_cloud
```


### Run CPU_monitor
Run CPU_monitor (You don't have to run it, it's just for monitoring the CPU and memory usage of our method)
```
roslaunch cpu_monitor cpu_monitor.launch poll_period:=1
```

Run python scripts to show CPU and memory usage of our nodes, cpu_mem2.py will print the current mean value of CPU and memory usage of our nodes. If you just use the original raw pointcloud directly to kiss-icp, please set: use_keypoint_pointcloud=False, inside the codes.

```
python cpu_mem2.py 
```

### Play the rosbag

```
rosbag play [rosbag that you have]
```

## Citation
If you find our work useful in your research, please consider citing:
```
@Article{rs15205074,
AUTHOR = {Zhang, Haizhou and Yu, Xianjia and Ha, Sier and Westerlund, Tomi},
TITLE = {LiDAR-Generated Images Derived Keypoints Assisted Point Cloud Registration Scheme in Odometry Estimation},
JOURNAL = {Remote Sensing},
VOLUME = {15},
YEAR = {2023},
NUMBER = {20},
ARTICLE-NUMBER = {5074},
URL = {https://www.mdpi.com/2072-4292/15/20/5074},
ISSN = {2072-4292},
DOI = {10.3390/rs15205074}
}
```

## License
Distributed under the MIT License. See LICENSE for more information.

## Contact
For any queries regarding the code or the paper, please open an issue in this repository.


