// #include "stdafx.h"
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/highgui/highgui.hpp>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#define PCL_NO_PRECOMPILE
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl_conversions/pcl_conversions.h>

#include <boost/thread/thread.hpp>
#include <chrono>
#include <queue>
#include <iostream>
#include <filesystem>

#include "superpoint/NetWork.hpp"
#include "superpoint/PointTracker.hpp"
#include "superpoint/SuperPointFrontend.hpp"


#define sensor_tpye "os0"

class lidarImageKeypointOdom
{
private:
    /* data */
    ros::NodeHandle nh_;

    ros::Timer timer_;

    ros::Subscriber pointCloudSub_;

    ros::Subscriber keyPointsSub_;

    ros::Subscriber signalImagesSub_;

    ros::Publisher keyPointCloudPub_;

    ros::Subscriber rangeImageSub_;

    ros::Subscriber imu_sub_;

    ros::Publisher imu_pub_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr entirePointCloudPtr_{new pcl::PointCloud<pcl::PointXYZ>};

    pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointsPtr_{new pcl::PointCloud<pcl::PointXYZ>};

    sensor_msgs::Imu last_received_imu_;

    // pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloudPtr_{new pcl::PointCloud<pcl::PointXYZ>};

    cv::Mat img_;
    cv::Mat rangeImg_;

    cv_bridge::CvImagePtr cvPtr_;

private:

    std::chrono::steady_clock::time_point pc_time_;

    std::chrono::steady_clock::time_point signal_time_;

    std::chrono::steady_clock::time_point range_time_;

    std::chrono::steady_clock::time_point timer_time_;

    int cnt_ = 0;

    std::queue<cv::Mat> img_queue_;

    std::string weight_path_ = "/home/hasar/dev/ws_livox/src/ws-lidar-as-camera-odom/src/signal-image-keypoints-odom/model/superpoint_v2.pt";
   
    SPFrontend spfrontend_{weight_path_, 4, 0.015, 0.4};

#ifdef sensor_tpye=="os0"
    int SH_ = 128;
    int SW_ = 2048;
    int initial_arr[4] = {12, 4, -4, -12};
    int pixel_shift_by_row_[128];
#else
    int SH_ = 64;
    int SW_ = 2048;
    int initial_arr[2] = {48, 16};
    int pixel_shift_by_row_[64];
#endif

public:

    lidarImageKeypointOdom(ros::NodeHandle *nh_);

    ~lidarImageKeypointOdom();

public:

    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);

    void keyPointsCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);

    void signalImageCallback(const sensor_msgs::ImageConstPtr& msg);

    void rangeImageCallback(const sensor_msgs::ImageConstPtr& msg);

    void timerCallback(const ros::TimerEvent& event);

    void superpointtimerCallback(const ros::TimerEvent& event);

    void imuCallback(const sensor_msgs::ImuConstPtr& imu);

public:

    void publishPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pc); 
    

};

