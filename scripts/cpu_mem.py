# std_msgs/Float32

import numpy as np
import rospy
from std_msgs.msg import Float32
from std_msgs.msg import UInt64
import time




lio_1_cpu = []
lio_2_cpu = []
lio_1_mem = []
lio_2_mem = []


i =0

length = 64

##-------------------------------------------------------lio-livox----------------------------------------------
def lio1_cpu_cb(data):
    # print('Going')
    lio_1_cpu.append(data.data)
    if len(lio_1_cpu) == length:
        np.save("./usage/kiss_cpu.npy", lio_1_cpu)
        print('done!')
def lio2_cpu_cb(data):
    lio_2_cpu.append(data.data)
    if len(lio_2_cpu) == length:
        np.save("./usage/kp_cpu.npy", lio_2_cpu)
        print('done!')
def lio1_mem_cb(data):
    lio_1_mem.append(data.data)
    if len(lio_1_mem) == length:
        np.save("./usage/kiss_mem.npy", lio_1_mem)
        print('done!')
def lio2_mem_cb(data):
    lio_2_mem.append(data.data)
    if len(lio_2_mem) == length:
        np.save("./usage/kp_mem.npy", lio_2_mem)
        print('done!')

#
def listener(): 
    rospy.init_node('listener', anonymous=True)


    #--------------------------------LIO-LIVOX--------------------------------
    rospy.Subscriber("/cpu_monitor/odometry_node/cpu", Float32, lio1_cpu_cb)               
    rospy.Subscriber("/cpu_monitor/odometry_node/mem", UInt64, lio1_mem_cb)  
    rospy.Subscriber("/cpu_monitor/signal_image_keypoints_odom_node/cpu", Float32, lio2_cpu_cb)               
    rospy.Subscriber("/cpu_monitor/signal_image_keypoints_odom_node/mem", UInt64, lio2_mem_cb)









    rospy.spin()

if __name__ == '__main__':           
    listener()