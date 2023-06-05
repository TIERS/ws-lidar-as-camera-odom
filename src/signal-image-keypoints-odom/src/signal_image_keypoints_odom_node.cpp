#include "signal_image_keypoints_odom/signal_image_keypoints_odom.hpp"

int main(int argc, char** argv)
{
    // Initialize the ROS node
    ros::init(argc, argv, "signal_image_keypoints_odom_node");

    ros::NodeHandle nh;

    lidarImageKeypointOdom lidarImgKeyOdom = lidarImageKeypointOdom(&nh);
    // Spin the node
    ros::spin();

    return 0;
}