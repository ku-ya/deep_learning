# !/usr/bin/env python3
import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.interpolate import interp1d
import time
import pdb
import rosbag
from sklearn.neighbors import NearestNeighbors
import math
import icp

# fname = '1487017469966396563'
def icp(a, b, init_pose=(0,0,0), no_iterations = 13):
    src = np.array([a.T], copy=True).astype(np.float32)
    dst = np.array([b.T], copy=True).astype(np.float32)
    #Initialise with the initial pose estimation
    Tr = np.array([[np.cos(init_pose[2]),-np.sin(init_pose[2]),init_pose[0]],
                   [np.sin(init_pose[2]), np.cos(init_pose[2]),init_pose[1]],
                   [0,                    0,                   1          ]])

    src = cv2.transform(src, Tr[0:2])
    for i in range(no_iterations):
        #Find the nearest neighbours between the current source and the
        #destination cloudpoint
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst[0])
        distances, indices = nbrs.kneighbors(src[0])
        #Compute the transformation between the current source
        #and destination cloudpoint
        T = cv2.estimateRigidTransform(src, dst[0, indices.T], False)
        #Transform the previous source and update the
        #current source cloudpoint
        src = cv2.transform(src, T)
        #Save the transformation from the actual source cloudpoint
        #to the destination
        Tr = np.dot(Tr, np.vstack((T,[0,0,1])))
    return Tr[0:2]

def laser(data, pose):
    r = data
    N = len(data)
    laser_angle_range = 270./180*np.pi
    angle_offset = -18./180*np.pi
    laser_angle = np.linspace(-1./2*laser_angle_range + pose[2],1./2*laser_angle_range + pose[2], N, endpoint=True)
    plt.plot(r*np.cos(laser_angle) + pose[0], r*np.sin(laser_angle) + pose[1], 'r.')
#
# bag = rosbag.Bag('/media/kuya/e7916bba-cd32-4e8e-b40c-c0705c6699f3/2017-02-13-15-24-28.bag')
#
# i = 0
# a = []
# b = []
# guess = [0.,0.,0.]
# z = np.empty([2, 1081])
# z_c = np.empty([2,1081])
# x = [0,0,0]
# plt.grid(True)
# axes = plt.gca()
# axes.set_xlim([-2, 2])
# axes.set_ylim([-2,2])
#
# for topic, msg, t in bag.read_messages(topics=['/scan']):
#     angle = np.linspace(msg.angle_min,msg.angle_max, 1081)
#     if i == 0:
#         z = np.array([msg.ranges*np.cos(angle),msg.ranges*np.sin(angle)])
#         # laser(msg.ranges,x)
#
#     else:
#         # plt.clf()
#         plt.grid(True)
#         axes = plt.gca()
#         axes.set_xlim([-4, 4])
#         axes.set_ylim([-4,4])
#         z_c = np.array([msg.ranges*np.cos(angle),msg.ranges*np.sin(angle)])
#         Tr = icp(z, z_c,guess, 300)
#         # plt.plot(x[0], x[1],'o')
#         # plt.draw()
#         # plt.pause(0.05)
#         x[0] = x[0] - Tr[0,2]
#         x[1] = x[1] - Tr[1,2]
#         x[2] = x[2] - math.atan2(Tr[1,0], Tr[0,0])
#         # guess =[Tr[0,2],  Tr[1,2], math.atan2(Tr[1,0], Tr[0,0])]
#         print Tr
#         print math.atan2(Tr[1,0], Tr[0,0])
#         # if i%10==0:
#         # pdb.set_trace()
#         src = np.array([z.T]).astype(np.float32)
#         res = cv2.transform(src, Tr)
#         # plt.plot(x[0],x[1],'rx')
#         # laser(msg.ranges, x)
#         # plt.pause(0.1)
#         # plt.plot(z_c[0],z_c[1],'b')
#         # plt.plot(res[0].T[0], res[0].T[1], 'r.')
#
#         src = np.array([z_c.T]).astype(np.float32)
#         Tr[0,0] = np.cos(x[2])
#         Tr[0,1] = -np.sin(x[2])
#         Tr[1,0] = np.sin(x[2])
#         Tr[1,1] = np.cos(x[2])
#         Tr[0,2] = x[0]
#         Tr[1,2] = x[1]
#         res = cv2.transform(src,Tr)
#
#         plt.plot(res[0].T[0], res[0].T[1], 'm.')
#         # plt.plot(z[0], z[1],'g')
#         # plt.show()
#         plt.pause(0.05)
#         z = z_c
#
#     if i > 500:
#         break
#     i = i + 1
#
# plt.show
# sys.exit()


# ang = np.linspace(-np.pi/2, np.pi/2, 320)
# a = np.array([ang, np.sin(ang)])
# th = np.pi/2
# rot = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
# b = np.dot(rot, a) + np.array([[0.2], [0.3]])

# M2 = icp(a, b, [0,  0, 0], 40)
# print M2
# src = np.array([a.T]).astype(np.float32)
# res = cv2.transform(src, M2)
# pdb.set_trace()

# plt.figure()
# plt.plot(b[0],b[1])
# plt.plot(res[0].T[0], res[0].T[1], 'r.')
# plt.plot(a[0], a[1])
# plt.show()


# bag.close()
# sys.exit()

data_path = 'data/sensor_pos_data'
# r = np.load(data_path+'/laser/'+fname+'.npy')
# N = len(r)



# f = interp1d(laser_angle, x[::-1],kind='cubic')
# plt.plot(angle, x[::-1],'o')
# x = np.load(data_path+'/depth/'+fname+'.npy')
def depth(data, pose):
    r = np.mean(data,axis=0)
    N = 640
    angle_range = 58./180*np.pi
    angle = np.linspace(1./2*angle_range,-1./2*angle_range,N, endpoint=True)
    plt.plot(pose[0] + r*np.cos(angle + pose[2]),pose[1] + r*np.sin(angle + pose[2]),'g.')
plt.grid(True)
axes = plt.gca()
axes.set_xlim([-10, 10])
axes.set_ylim([-10,10])
# print(f(angle).shape)
# plt.plot(angle,f(angle)-x, 'o')
# pylab.xlim([-1./2*angle_range, 1./2*angle_range])
# pylab.ylim([0, 10])
# plt.show()

laser_dir = data_path + '/laser/'
pose_dir = data_path + '/pose/'
depth_dir = data_path + '/depth/'
# out_directory = data_path + '/laser_fov/'
i = 0
for filename in os.listdir(laser_dir):
    if filename.endswith(".npy"):
        # print(os.path.join(directory, filename))
        # print(filename)
        laser_in = np.load(laser_dir+filename)
        pose_in = np.load(pose_dir+filename)
        depth_in = np.load(depth_dir+filename)
        # depth_in[np.isnan(depth_in)] = 10
        # print depth_in
        laser(laser_in, pose_in)
        depth(depth_in, pose_in)
        # plt.plot([pose_in[0], pose_in[0]+np.cos(pose_in[2])],[pose_in[1], pose_in[1]+np.sin(pose_in[2])], '-o')
        # plt.draw()
        # print(pose_in)
        plt.pause(0.05)
        print pose_in
        i = i + 1
        if i > 0:
            break
            # plt.close()
        # f = interp1d(laser_angle, depth_in[::-1],kind='linear')
        # data_out = f(angle)
        # np.save(out_directory+filename,data_out)
        # continue
    # else:
    #     continue

# plt.close()
