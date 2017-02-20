#!/usr/bin/env python
import roslib
# roslib.load_manifest('odroid')
import sys
import rospy
import cv2
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose2D
import numpy as np
import tf
import os
counter = 0
# from __future__ import print_function
class image_converter:
  def __init__(self):
    self.image_pub = rospy.Publisher("depth_reduced",Image,queue_size=100)
    self.bridge = CvBridge()
    depth_sub = message_filters.Subscriber("/camera/depth/image_raw",Image)
    rgb_sub = message_filters.Subscriber("/camera/rgb/image_raw",Image)
    laser_sub = message_filters.Subscriber("/scan",LaserScan)
    ir_sub = message_filters.Subscriber("/camera/ir/image",Image)


    self.ts = message_filters.ApproximateTimeSynchronizer([depth_sub, rgb_sub, laser_sub], 10, 0.5, allow_headerless=False)
    self.ts.registerCallback(self.callback)


  def callback(self,depth_data, rgb_data, laser_data):
    global counter
    try:
      rgb_image = self.bridge.imgmsg_to_cv2(rgb_data, "passthrough")
      depth_image = self.bridge.imgmsg_to_cv2(depth_data, "passthrough")
      # ir_image = self.bridge.imgmsg_to_cv2(ir_data, "passthrough")
    except CvBridgeError as e:
      print(e)

    # (rows,cols) = cv_image.shape
    # print(rows, cols)
    # image[235:245,:,0] = 0
    # image[235:245,:,1] = 0
    # image[235:245,:,2] = cv_image[235:245,:]

    # if cols > 60 and rows > 60 :
    #   cv2.circle(depth_image, (50,50), 10, 255)

    # cv_image = cv2.resize(cv_image, (0,0), fx=0.2, fy=0.2)

    # cv2.imshow("Image window", cv_image)
    # cv2.imshow("Image window", image)
    # cv2.waitKey(10)
    # listener = tf.TransformListener()

    # (trans,rot) = listener.lookupTransform('/world', '/laser', rospy.Time(0))
    # pose = np.array([pose_data.x,pose_data.y,pose_data.theta])
    # print(pose_data



    try:
      fname =str(depth_data.header.stamp)
      cv2.imwrite('rgb/'+fname+'.jpg',rgb_image)
      np.save('depth/'+fname, depth_image)
      np.save('laser/'+fname, laser_data.ranges)
      # np.save('ir/'+fname,ir_image)
      print('saving data:' + str(counter))
      counter=counter+1
    except CvBridgeError as e:
      print(e)

    # try:
      # self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "passthrough"))
    # except CvBridgeError as e:
    #   print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_resize', anonymous=True)
  cwd = os.getcwd()
  print cwd
  directory = cwd+'/rgb'
  if not os.path.exists(directory):
    os.makedirs(directory)
  directory = cwd+'/laser'
  if not os.path.exists(directory):
    os.makedirs(directory)
  directory = cwd+'/depth'
  if not os.path.exists(directory):
    os.makedirs(directory)
  directory = cwd+'/ir'
  if not os.path.exists(directory):
    os.makedirs(directory)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
