#!/usr/bin/python
import os, sys
current_dir = os.getcwd()
from utils import *
from keras.layers import Merge
import time
# import imageio
from keras.models import model_from_json
import bcolz
import pdb
import tensorflow as tf
import pandas as pd
from keras.callbacks import LearningRateScheduler

class visualization(object):
    """docstring for visualization."""
    def __init__(self):
        super(visualization, self).__init__()

    def depth_plot(self, data, pose):
        r = np.mean(data.T,axis=0)
        N = len(r)
        resol = 0.0906
        angle_range = resol*N/180*np.pi
        angle = np.linspace(-1./2*angle_range, 1./2*angle_range,N, endpoint=True) - pose[2]
        plt.plot(- pose[0] + r*np.cos(angle), - pose[1] + r*np.sin(angle),'g')

    def laser_plot(self, data, pose):
        r = data
        N = len(data)
        resol = 0.25
        laser_angle_range = resol*N/180*np.pi
        angle_offset = -18./180*np.pi
        laser_angle = np.linspace(-1./2*laser_angle_range, 1./2*laser_angle_range, N, endpoint=True) - pose[2]
        plt.plot(r*np.cos(laser_angle) - pose[0], r*np.sin(laser_angle) - pose[1],'b')

    def laser_plot2(self, data, pose):
        r = data
        N = len(data)
        resol = 0.25
        laser_angle_range = resol*N/180*np.pi
        angle_offset = -18./180*np.pi
        laser_angle = np.linspace(-1./2*laser_angle_range, 1./2*laser_angle_range, N, endpoint=True) - pose[2]
        plt.plot(r*np.cos(laser_angle) - pose[0], r*np.sin(laser_angle) - pose[1], 'r')

class data_handler(object):
    """docstring for data_handler."""
    def __init__(self):
        super(data_handler, self).__init__()

    def load(self, data_path='data/sensor_pos_data/', pose_fname='pose.dat', laser_fname='laser.dat', rgb_fname='rgb.dat', depth_fname='depth.dat'):
        pose = None
        laser = None
        rgb = None
        depth = None
        try:
            pose = load_array(data_path+'pose.dat')
            laser = load_array(data_path+'laser.dat')
            rgb = load_array(data_path+'rgb.dat')
            depth = load_array(data_path+'depth.dat')
            depth = depth[..., None]
            pose = pose[...,None]
            return (pose, laser, rgb, depth, True)
        except Exception as e:
            print "Exception caught:", e
            return (pose, laser, rgb, depth, False)

class keras_model(object):
    """docstring for keras_model."""
    def __init__(self, output_shape):
        super(keras_model, self).__init__()
        self.pose_model = None
        self.conv_model = None
        self.depth_model = None
        self.final_model = None
        self.output_shape = output_shape
        self.sgd = SGD(lr=0.0, decay=1e-4, momentum=0.9, nesterov=True)
        self.tbCallback=keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
        self.lrate = LearningRateScheduler(self.step_decay)
        self.init_model()

    def init_model(self):
        self.pose_model = self.create_pose_model()
        self.conv_model = self.create_conv_model()
        self.depth_model = self.create_depth_model()
        self.final_model = self.create_final_model(self.depth_model, self.conv_model, self.pose_model)

    def create_pose_model(self):
        pose_model = Sequential()
        # pose_model.add(BatchNormalization(axis=1,input_shape=(3,1)))
        pose_model.add(Dense(1, input_shape=(3,1), activation='linear'))
        pose_model.add(Flatten())
        return pose_model

    def create_depth_model(self):
        depth_model = Sequential()
        depth_model.add(BatchNormalization(axis=1,input_shape=(10, 640,1)))
        depth_model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu',dim_ordering = 'tf'))
        depth_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        depth_model.add(BatchNormalization())
        depth_model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu',dim_ordering = 'tf'))
        depth_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        depth_model.add(BatchNormalization())
        depth_model.add(Convolution2D(64, 1, 3, border_mode='same', activation='relu',dim_ordering = 'tf'))
        depth_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        depth_model.add(BatchNormalization())
        depth_model.add(Convolution2D(64, 1, 3, border_mode='same', activation='relu',dim_ordering = 'tf'))
        depth_model.add(MaxPooling2D((1, 2), strides=(1, 2)))
        depth_model.add(BatchNormalization())
        # depth_model.add(Dense(1, activation='relu'))
        depth_model.add(Flatten())
        return depth_model

    def create_conv_model(self):
        model = Sequential()
        model.add(BatchNormalization(axis=1,input_shape=(10, 640,3)))
        model.add(Convolution2D(32, 3, 3, border_mode='same', dim_ordering = 'tf',activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering = 'tf'))
        model.add(BatchNormalization())
        model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', dim_ordering = 'tf' ))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering = 'tf' ))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering = 'tf' ))
        model.add(MaxPooling2D((1, 2), strides=(1, 2)))
        model.add(BatchNormalization())
        model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering = 'tf' ))
        model.add(MaxPooling2D((1, 2), strides=(1, 2)))
        model.add(BatchNormalization())
        model.add(Flatten())
        return model

    def create_final_model(self, depth_model, model, pose_model):
        final_model = Sequential()
        merge = Merge([depth_model, model, pose_model],mode='concat')
        final_model.add(merge)
        # final_model.add(BatchNormalization())
        # final_model.add(Dense(10000, activation='relu'))
        final_model.add(Dense(self.output_shape, activation='linear'))
        final_model.summary()
        return final_model

    def step_decay(self,epoch):
    	initial_lrate = 0.1
    	drop = 0.5
    	epochs_drop = 10.0
    	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    	return lrate

    def mean_squared_error_exp(self, y_true, y_pred):
        return K.mean(K.square(y_pred - y_true)*tf.exp(-tf.abs(y_true)/(1.5**2)), axis=-1)

    def compile(self):
        self.final_model.compile(loss=self.mean_squared_error_exp, optimizer=self.sgd)

if __name__ == "__main__":
    pass
