#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

# def callback(data):
#     # point_cloud_size = data.size()
#     rospy.loginfo("Received point cloud width and height: %d and %d", data.width, data.height)


def callback(data):
    rospy.loginfo("Received metadata: %s", data.data)

def listener():
    rospy.init_node('metadata_subscriber', anonymous=True)
    rospy.Subscriber("/ouster/metadata", String, callback)    
    # rospy.Subscriber("/os0_cloud_node/points", PointCloud2, callback)
    rospy.spin()


    
if __name__ == '__main__':
    listener()
    # l = [
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16,
    #         48,
    #         16
    #     ]
    # print(len(l))