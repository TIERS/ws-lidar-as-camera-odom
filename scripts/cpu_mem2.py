import numpy as np
import rospy
from std_msgs.msg import Float32
from std_msgs.msg import UInt64
import time

odometry_node_cpu_data = None
odometry_node_mem_data = None
keypoints_node_cpu_data = None
keypoints_node_mem_data = None

odometry_node_cpu_list = []
odometry_node_mem_list = []
keypoints_node_cpu_list = []
keypoints_node_mem_list = []

use_keypoint_pointcloud=True



def odometry_node_cpu_cb(data):
    global odometry_node_cpu_data
    odometry_node_cpu_data = data.data
    if not use_keypoint_pointcloud:
        if odometry_node_cpu_data > 2.0 :
            odometry_node_cpu_list.append(odometry_node_cpu_data)
            odometry_node_mem_list.append(odometry_node_mem_data)
            print("-------------------------------")
            print_data_only_odometry()

        else:
            print('Using the raw pointcloud, but not data fed in yet')

    
def odometry_node_mem_cb(data):
    global odometry_node_mem_data
    odometry_node_mem_data = data.data



def keypoints_node_cpu_cb(data):
    global keypoints_node_cpu_data
    keypoints_node_cpu_data = data.data

    global use_keypoint_pointcloud
    if use_keypoint_pointcloud :
        if keypoints_node_cpu_data > 10 and odometry_node_cpu_data > 2.0:
            odometry_node_cpu_list.append(odometry_node_cpu_data)
            keypoints_node_cpu_list.append(keypoints_node_cpu_data)
            odometry_node_mem_list.append(odometry_node_mem_data)
            keypoints_node_mem_list.append(keypoints_node_mem_data)
            print("-------------------------------")
            print_data()
        else:
            print("Using the pointcloud from keypoint, but not data fed in yet")

def keypoints_node_mem_cb(data):
    global keypoints_node_mem_data
    keypoints_node_mem_data = data.data

def print_data():
    print('Mean cpu:', np.mean(odometry_node_cpu_list)+np.mean(keypoints_node_cpu_list))
    print('Mean memory:', np.mean(odometry_node_mem_list)+np.mean(keypoints_node_mem_list))

def print_data_only_odometry():
    print('Mean cpu:', np.mean(odometry_node_cpu_list))
    print('Mean memory:', np.mean(odometry_node_mem_list))


def listener(): 
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/cpu_monitor/odometry_node/cpu", Float32, odometry_node_cpu_cb)               
    rospy.Subscriber("/cpu_monitor/odometry_node/mem", UInt64, odometry_node_mem_cb)  
    rospy.Subscriber("/cpu_monitor/signal_image_keypoints_odom_node/cpu", Float32, keypoints_node_cpu_cb)               
    rospy.Subscriber("/cpu_monitor/signal_image_keypoints_odom_node/mem", UInt64, keypoints_node_mem_cb)

    rospy.spin()

if __name__ == '__main__':           
    listener()
