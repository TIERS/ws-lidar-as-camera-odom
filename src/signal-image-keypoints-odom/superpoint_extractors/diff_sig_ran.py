#!/usr/bin/env python

import argparse
import glob
import numpy as np
import os
import time
import cv2
import torch
from sklearn.neighbors import NearestNeighbors
import csv
import rospy
import ros_numpy
import math
from sensor_msgs.msg import Image, PointCloud2, ChannelFloat32
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointField
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Point32

from superpoint_keypoint import SuperPointFrontend, SuperPointNet

import pcl

class VideoStreamer(object):
  """ 
    Class to help process image streams. Three types of possible inputs:"
    1.) USB Webcam.
    2.) A directory of images (files in directory matching 'img_glob').
    3.) A video file, such as an .mp4 or .avi file.
  """
  def __init__(self, basedir, camid, height, width, skip, img_glob):
    self.cap = []                    ################################
    self.camera = False              ################################
    self.video_file = False          ################################
    self.listing = []                ################################
    self.sizer = [height, width]     ################################
    self.i = 0                       ################################
    self.skip = skip                 ################################
    self.maxlen = 1000000            ################################
    # If the "basedir" string is the word camera, then use a webcam.
    if basedir == "camera/" or basedir == "camera":
      print('==> Processing Webcam Input.')
      self.cap = cv2.VideoCapture(camid)
      self.listing = range(0, self.maxlen)
      self.camera = True
    else:
      # Try to open as a video.
      self.cap = cv2.VideoCapture(basedir)
      lastbit = basedir[-4:len(basedir)]
      if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
        raise IOError('Cannot open movie file')
      elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
        print('==> Processing Video Input.')
        num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.listing = range(0, num_frames)
        self.listing = self.listing[::self.skip]
        self.camera = True
        self.video_file = True
        self.maxlen = len(self.listing)
      else:
        print('==> Processing Image Directory Input.')   ################################
        search = os.path.join(basedir, img_glob)         ################################
        self.listing = glob.glob(search)                 ################################
        self.listing.sort()                              ################################
        self.listing = self.listing[::self.skip]         ################################
        self.maxlen = len(self.listing)                  ################################
        if self.maxlen == 0:                             ################################
          raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')    ################################

  def read_image(self, impath, img_size):
    """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
      img_size: (W, H) tuple specifying resize size.
    Returns
      grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    grayim = cv2.imread(impath, 0)
    if grayim is None:
      raise Exception('Error reading image %s' % impath)
    # Image is resized via opencv.
    interp = cv2.INTER_AREA
    grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
    grayim = (grayim.astype('float32') / 255.)
    return grayim

  def next_frame(self):
    """ Return the next frame, and increment internal counter.
    Returns
       image: Next H x W image.
       status: True or False depending whether image was loaded.
    """
    if self.i == self.maxlen:
      return (None, False)
    if self.camera:
      ret, input_image = self.cap.read()
      if ret is False:
        print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
        return (None, False)
      if self.video_file:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
      input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                               interpolation=cv2.INTER_AREA)
      input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
      input_image = input_image.astype('float')/255.0
    else:
      image_file = self.listing[self.i]                             #####################################
      input_image = self.read_image(image_file, self.sizer)         #####################################
    # Increment internal counter.
    self.i = self.i + 1                                             #####################################
    input_image = input_image.astype('float32')                     #####################################
    return (input_image, True)                                      #####################################



class ImageProcessor:
    def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh, cuda):
        rospy.init_node('pointcloud_image_processor', anonymous=True)
        self.bridge = CvBridge()
        self.signal_frame = None
        self.range_frame = None
        self.new_image = False
        self.image_count = 0
        self.height = 128
        self.width = 2048
        self.pc_time = time.time()
        self.img_time = time.time()
        self.timer_time = time.time()
        self.signal_heatmap = None
        self.pixel_shift_by_row = [12, 4, -4, -12] * (self.height // 4)
        # print(self.pixel_shift_by_row)
        rospy.loginfo('==> Loading pre-trained network.')
        # This class runs the SuperPoint network and processes its outputs.
        self.fe = SuperPointFrontend(weights_path=weights_path,
                            nms_dist=nms_dist,
                            conf_thresh=conf_thresh,
                            nn_thresh=nn_thresh,
                            cuda=cuda)
        rospy.loginfo('==> Successfully loaded pre-trained network.')
        self.cloud = np.array([])

        self.signal_sub = rospy.Subscriber("/os0_img_node/signal_image", Image, self.signal_callback)
        self.signal_heatmap_pub = rospy.Publisher("/signal_heatmap", Image, queue_size=10)
        self.range_sub = rospy.Subscriber("/os0_img_node/range_image", Image, self.range_callback)
        self.range_heatmap_pub = rospy.Publisher("/range_heatmap", Image, queue_size=10)
        # rospy.Timer(rospy.Duration(0.1), self.timer_callback)

    def signal_callback(self, data):
        self.image_count = self.image_count + 1
        # print("{}_call_back ".format(self.image_count))

        try:
        # because in the rosbag, the image is mono16 type
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='mono16') 
            
            # Convert the "mono16" image to a grayscale OpenCV image
            cv_image = np.uint8(cv_image / 256)
            ########################
            cv_image = cv2.resize(cv_image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC) 
            #######################

            # convert it to the input type of Superpoint network: HxW numpy float32 input image in range [0,1]
            cv_image = (cv_image.astype('float32') / 255.)
            cv_image = cv_image.astype('float32')

            self.signal_frame = cv_image
            signal_pts, desc, heatmap = self.fe.run(self.signal_frame)
            heatmap_signal = (heatmap * 255.0).astype(np.uint8)  # Convert to 0-255 range
            self.signal_heatmap = heatmap_signal
            heatmap_msg = self.bridge.cv2_to_imgmsg(heatmap_signal, encoding="mono8")
            self.signal_heatmap_pub.publish(heatmap_msg)


        except Exception as e:
            rospy.logerr("Error converting Image message: {}".format(e))

    
    def range_callback(self, data):
        self.image_count = self.image_count + 1
        # print("{}_call_back ".format(self.image_count))

        # try:
        # because in the rosbag, the image is mono16 type
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='mono16') 
        
        # cv_image = cv2.equalizeHist(cv_image)



        # cv_image = cv2.add(cv_image, 9000) 
        # heatmap_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="mono16")
        # self.range_heatmap_pub.publish(heatmap_msg)
        # Convert the "mono16" image to a grayscale OpenCV image
        cv_image = np.uint8(cv_image / 256)
        # cv_image = cv2.equalizeHist(cv_image)
        gamma = 0.5  # change the value here
        cv_image = np.array(255*(cv_image / 255) ** gamma, dtype = 'uint8') 
        ########################
        cv_image = cv2.resize(cv_image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC) 
        #######################

        # convert it to the input type of Superpoint network: HxW numpy float32 input image in range [0,1]
        cv_image = (cv_image.astype('float32') / 255.)
        cv_image = cv_image.astype('float32')

        self.range_frame = cv_image
        range_pts, desc, range_heatmap = self.fe.run(self.range_frame)
        heatmap_range = (range_heatmap * 255.0).astype(np.uint8)  # Convert to 0-255 range
        
        red_heatmap = np.zeros((self.signal_heatmap.shape[0], self.signal_heatmap.shape[1], 3), dtype=np.uint8)
        red_heatmap[:, :, 2] = self.signal_heatmap

        # Convert heatmap2 to all green
        green_heatmap = np.zeros((heatmap_range.shape[0], heatmap_range.shape[1], 3), dtype=np.uint8)
        green_heatmap[:, :, 1] = heatmap_range
        
        combined = cv2.addWeighted(red_heatmap, 1, green_heatmap, 1, 0)

        
        
        heatmap_msg = self.bridge.cv2_to_imgmsg(combined, encoding="bgr8")
        self.range_heatmap_pub.publish(heatmap_msg)


        # except Exception as e:
        #     rospy.logerr("Error converting Image message: {}".format(e))

  

  
#   def timer_callback(self, event):
#     # Process both PointCloud and Image messages here
#     # rospy.loginfo("Timer callback")
#     self.timer_time = time.time()
#     if(self.cloud.shape[0] > 0 and self.current_frame.size >0 and math.fabs(self.timer_time - self.pc_time) < 0.2 and  math.fabs(self.timer_time - self.img_time) < 0.2):
#       # pass
#       start1 = time.time()
#       pts, desc, heatmap = self.fe.run(self.current_frame)
#       end1 = time.time()
#       # Create PointCloud message and publish points
#       msg_cloud = PointCloud2()
#       msg_cloud.header.stamp = rospy.Time.now()
#       # msg_cloud.frame_id = self.cloud.frame_id
#       msg_cloud.header.frame_id = "keypoint_frame"
#       msg_cloud.width = self.width
#       msg_cloud.height = self.height
#       msg_cloud.data = [0] * (self.width * self.height)
#       msg_cloud.fields = [
#             PointField('x', 0, PointField.FLOAT32, 1),
#             PointField('y', 4, PointField.FLOAT32, 1),
#             PointField('z', 8, PointField.FLOAT32, 1)]
#       temp = np.zeros((self.width, self.height))
#       # a = np.asarray(self.cloud) 
#       for i in range(pts.shape[1]):
#           x, y, _= pts[:, i]
#           ####################################
#           x,y = int(x)*2, int(y)*2# now x y are the coordinates of the original image
#           ####################################
#           if(x>self.width and y>self.height):
#             continue
#           inx = int((x + self.width - self.pixel_shift_by_row[y]) % self.width)
#           # cloud.points.append(pt)
#           msg_cloud.data[inx] = self.cloud[inx]
#           # msg_cloud.data.append(self.cloud.data[inx])
#       # msg_cloud.data = np.asarray(np.array(temp), np.float32).tobytes()
#       self.points_pub.publish(msg_cloud)
#       rospy.loginfo("Point Cloud Published!")
#     # pass

    def run(self):
        rospy.spin()


if __name__ == '__main__':

  # Parse command line arguments.
  parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
  #parser.add_argument('--input', type=str, default='',
      #help='Image directory or movie file or "camera" (for webcam).')
  parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
      help='Path to pretrained weights file (default: superpoint_v1.pth).')
  parser.add_argument('--img_glob', type=str, default='*.png',
      help='Glob match if directory of images is specified (default: \'*.png\').')
  parser.add_argument('--skip', type=int, default=1,
      help='Images to skip if input is movie or directory (default: 1).')
  parser.add_argument('--show_extra', action='store_true',
      help='Show extra debug outputs (default: False).')
  parser.add_argument('--H', type=int, default=120,
      help='Input image height (default: 120).')
  parser.add_argument('--W', type=int, default=160,
      help='Input image width (default:160).')
  parser.add_argument('--display_scale', type=int, default=2,
      help='Factor to scale output visualization (default: 2).')
  parser.add_argument('--min_length', type=int, default=2,
      help='Minimum length of point tracks (default: 2).')
  parser.add_argument('--max_length', type=int, default=5,
      help='Maximum length of point tracks (default: 5).')
  parser.add_argument('--nms_dist', type=int, default=4,
      help='Non Maximum Suppression (NMS) distance (default: 4).')
  parser.add_argument('--conf_thresh', type=float, default=0.015,
      help='Detector confidence threshold (default: 0.015).')
  parser.add_argument('--nn_thresh', type=float, default=0.7,
      help='Descriptor matching threshold (default: 0.7).')
  parser.add_argument('--camid', type=int, default=0,
      help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
  parser.add_argument('--waitkey', type=int, default=1,
      help='OpenCV waitkey time in ms (default: 1).')
  parser.add_argument('--cuda', action='store_true',
      help='Use cuda GPU to speed up network processing speed (default: False)')
  parser.add_argument('--no_display', action='store_true',
      help='Do not display images to screen. Useful if running remotely (default: False).')
  parser.add_argument('--write', action='store_true',
      help='Save output frames to a directory (default: False)')
  parser.add_argument('--write_dir', type=str, default='tracker_outputs/',
      help='Directory where to write output frames (default: tracker_outputs/).')
  opt = parser.parse_args()
  print(opt)

  image_number = 1

  # rospy.init_node('image_processor', anonymous=True)
  image_processor = ImageProcessor(opt.weights_path, opt.nms_dist, opt.conf_thresh, opt.nn_thresh, opt.cuda)
  image_processor.run()
  # print('==> Waiting for image.')
  ###############################################

  # while not rospy.is_shutdown():
  #   if image_processor.new_image:
  #     img = image_processor.current_frame
    
  #     image_processor.new_image = False  # Reset the flag

  #     # Get points and descriptors.
  #     start1 = time.time()
  #     pts, desc, heatmap = fe.run(img)
  #     end1 = time.time()

  #     # Create PointCloud message and publish points
  #     cloud = PointCloud()
  #     cloud.header.stamp = rospy.Time.now()
  #     cloud.header.frame_id = "from image "+ str(image_number)  
  #     for i in range(pts.shape[1]):
  #         pt = Point32()
  #         pt.x, pt.y, _= pts[:, i]
  #         pt.z = 0  # Ignore z coordinate
  #         cloud.points.append(pt)

  #     image_processor.points_pub.publish(cloud)

  #     print("Already Processed image: {}, Time taken by detectors and descriptors: {:.4f} ms".format(image_number, (end1 - start1)*1000))

  #     image_number =  image_number + 1