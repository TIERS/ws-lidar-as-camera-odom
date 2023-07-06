// #include "stdafx.h"
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/Image.h>


#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/core.hpp>
// #include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
// #include <opencv2/xfeatures2d.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/xfeatures2d.hpp>
// #include <opencv2/xfeatures2d/nonfree.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#define PCL_NO_PRECOMPILE
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/filters/extract_indices.h>
// #include <pcl/kdtree/kdtree.h>
// #include <pcl-1.10/pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/voxel_grid.h>
// #include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

#include <chrono>
#include <queue>


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

    pcl::PointCloud<pcl::PointXYZ>::Ptr entirePointCloudPtr_{new pcl::PointCloud<pcl::PointXYZ>};

    pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointsPtr_{new pcl::PointCloud<pcl::PointXYZ>};

    // pcl::PointCloud<pcl::PointXYZ>::Ptr keyPointCloudPtr_{new pcl::PointCloud<pcl::PointXYZ>};

    cv::Mat img_;

    cv_bridge::CvImagePtr cvPtr_;

private:

    std::chrono::steady_clock::time_point pc_time_;

    std::chrono::steady_clock::time_point img_time_;

    std::chrono::steady_clock::time_point timer_time_;

    int cnt_ = 0;

    std::queue<cv::Mat> img_queue_;

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

    void timerCallback(const ros::TimerEvent& event);

public:

    void publishPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pc); 
    

};

