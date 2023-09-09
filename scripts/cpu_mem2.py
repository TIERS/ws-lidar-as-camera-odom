import numpy as np
import rospy
from std_msgs.msg import Float32
from std_msgs.msg import UInt64
import time

lio_1_cpu_data = None
lio_2_cpu_data = None
lio_1_mem_data = None
lio_2_mem_data = None

lio_1_cpu_list = []
lio_2_cpu_list = []
lio_1_mem_list = []
lio_2_mem_list = []

def lio1_cpu_cb(data):
    global lio_1_cpu_data
    lio_1_cpu_data = data.data

def lio1_mem_cb(data):
    global lio_1_mem_data
    lio_1_mem_data = data.data

def lio2_cpu_cb(data):
    global lio_2_cpu_data
    lio_2_cpu_data = data.data
    if lio_2_cpu_data > 10 and lio_1_cpu_data > 3:
        lio_1_cpu_list.append(lio_1_cpu_data)
        lio_2_cpu_list.append(lio_2_cpu_data)
        lio_1_mem_list.append(lio_1_mem_data)
        lio_2_mem_list.append(lio_2_mem_data)
        print_data()
    else:
        print('Not active yet')

def lio2_mem_cb(data):
    global lio_2_mem_data
    lio_2_mem_data = data.data

def print_data():
    print('Mean /cpu_monitor/odometry_node/cpu:', np.mean(lio_1_cpu_list))
    print('Mean /cpu_monitor/odometry_node/mem:', np.mean(lio_1_mem_list))
    print('Mean /cpu_monitor/signal_image_keypoints_odom_node/cpu:', np.mean(lio_2_cpu_list))
    print('Mean /cpu_monitor/signal_image_keypoints_odom_node/mem:', np.mean(lio_2_mem_list))

def listener(): 
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/cpu_monitor/odometry_node/cpu", Float32, lio1_cpu_cb)               
    rospy.Subscriber("/cpu_monitor/odometry_node/mem", UInt64, lio1_mem_cb)  
    rospy.Subscriber("/cpu_monitor/signal_image_keypoints_odom_node/cpu", Float32, lio2_cpu_cb)               
    rospy.Subscriber("/cpu_monitor/signal_image_keypoints_odom_node/mem", UInt64, lio2_mem_cb)

    rospy.spin()

if __name__ == '__main__':           
    listener()
