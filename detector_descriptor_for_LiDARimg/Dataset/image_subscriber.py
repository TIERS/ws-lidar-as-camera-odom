#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

image_count = 0

def image_callback(msg):
    global image_count
    print("Received an image!_{} ".format(image_count))
    # Convert the ROS image message to a NumPy array
    bridge = CvBridge()
    mono16_image = bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')

    # Convert the "mono16" image to a grayscale OpenCV image
    gray_image = np.uint8(mono16_image / 256)

    # Save the grayscale image to disk with a unique filename
    filename = f'./indoor/signal_image/image_{image_count:04d}.png'

    cv2.imwrite(filename, gray_image)
    
    # Increment the image counter
    image_count += 1

def image_subscriber():
	# ROS节点初始化
    rospy.init_node('image_subscriber')

	# 创建一个Subscriber，订阅名为/turtle1/pose的topic，注册回调函数poseCallback
    rospy.Subscriber('/img_node/signal_image', Image, image_callback)

	# 循环等待回调函数
    rospy.spin()

if __name__ == '__main__':
    image_subscriber()