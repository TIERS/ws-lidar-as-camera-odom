#ifndef SIGNAL_IMAGE_KEYPOINTS_ODOM_H
#define SIGNAL_IMAGE_KEYPOINTS_ODOM_H

// #include "stdafx.h"
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PointStamped.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
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

#define PCL_NO_PRECOMPILE

#define OS0 1
#define OTHER_SENSOR 2
#define sensor_type OS0//

class lidarImageKeypointOdom
{
private:
    /* data */
    ros::NodeHandle nh_;

    ros::Timer timer_;

    ros::Subscriber pointCloudSub_;
    ros::Subscriber rangeImageSub_;
    ros::Subscriber signalImagesSub_;
    ros::Publisher keyPointCloudPub_;
    ros::Subscriber imu_sub_;
    ros::Publisher imu_pub_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr entirePointCloudPtr_{new pcl::PointCloud<pcl::PointXYZ>};
    pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointsPtr_{new pcl::PointCloud<pcl::PointXYZ>};

    sensor_msgs::Imu last_received_imu_;



    cv::Mat signalImg_;
    cv::Mat rangeImg_;    

    std::chrono::steady_clock::time_point pc_time_;
    std::chrono::steady_clock::time_point signal_time_;
    std::chrono::steady_clock::time_point range_time_;
    std::chrono::steady_clock::time_point timer_time_;

    int cnt_ = 0;//so far, only used in traditional_method_callback
    std::queue<cv::Mat> img_queue_; //so far, only used in traditional_method_callback



    std::string weight_path_ = "/home/jimmy/Downloads/Performance_comparison/ws-lidar-as-camera-odom/src/signal-image-keypoints-odom/model/superpoint_v2.pt";
    SPFrontend spfrontend_{weight_path_, 4, 0.015, 0.4};


#if sensor_type == OS0
    int SH_ = 128;
    int SW_ = 2048;
#else
    int SH_ = 64;
    int SW_ = 2048;
#endif


public:

    lidarImageKeypointOdom(ros::NodeHandle *nh_);
    ~lidarImageKeypointOdom();

    void signalImageCallback(const sensor_msgs::ImageConstPtr& msg);
    void rangeImageCallback(const sensor_msgs::ImageConstPtr& msg);
    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);

    void traditional_method_timerCallback(const ros::TimerEvent& event);
    void superpoint_timerCallback(const ros::TimerEvent& event);
    void imuCallback(const sensor_msgs::ImuConstPtr& imu);
    void publishPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pc); 
    
    std::vector<cv::KeyPoint> extract_neighbourhood(const cv::Mat& img, const cv::Point& point, int size = 9, int threshold = 150);


    
};

#endif