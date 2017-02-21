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
        resol = 58./N
        angle_range = resol*N/180*np.pi
        angle = np.linspace(-1./2*angle_range, 1./2*angle_range,N, endpoint=True) - pose[2]
        plt.plot(- pose[0] + r*np.cos(angle), - pose[1] + r*np.sin(angle),'g')

    def laser_plot(self, data, pose):
        r = data
        N = len(data[0])
        resol = 0.25
        resol = 58./N
        laser_angle_range = resol*N/180*np.pi
        # angle_offset = -18./180*np.pi
        laser_angle = np.linspace(-1./2*laser_angle_range, 1./2*laser_angle_range, N, endpoint=True) - pose[2]
        plt.plot(r*np.cos(laser_angle) - pose[0], r*np.sin(laser_angle) - pose[1],'b')

    def laser_plot2(self, data, N_in, style):
        r = data
        N = N_in
        resol = 0.25
        resol = 58./N
        laser_angle_range = resol*N/180*np.pi
        # angle_offset = -18./180*np.pi
        laser_angle = np.linspace(-1./2*laser_angle_range, 1./2*laser_angle_range, N, endpoint=True)
        plt.plot(r*np.cos(laser_angle) , r*np.sin(laser_angle) , style)

class data_handler(object):
    """docstring for data_handler."""
    def __init__(self):
        super(data_handler, self).__init__()

    def load(self, data_path='data/sensor/', laser_fname='laser.dat', rgb_fname='rgb.dat', depth_fname='depth.dat'):
        laser = None
        rgb = None
        depth = None
        try:
            laser = load_array(data_path+'laser.dat')
            rgb = load_array(data_path+'rgb.dat')
            depth = load_array(data_path+'depth.dat')
            depth = depth[..., None]
            return (laser, rgb, depth, True)
        except Exception as e:
            print "Exception caught:", e
            return (laser, rgb, depth, False)

class keras_model(object):
    """docstring for keras_model."""
    def __init__(self, output_shape):
        super(keras_model, self).__init__()
        self.conv_model = None
        self.depth_model = None
        self.depth_model_merge = None
        self.final_model = None
        self.output_shape = output_shape
        self.sgd = SGD(lr=0.0, decay=1e-4, momentum=0.9, nesterov=True)
        self.tbCallback=keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
        self.lrate = LearningRateScheduler(self.step_decay)
        self.init_model()

    def init_model(self):
        self.conv_model = self.create_conv_model()
        self.depth_model = self.create_depth_model()
        self.final_model = self.create_final_model(self.depth_model, self.conv_model)

    def create_depth_model(self):
        depth_model = Sequential()
        depth_model.add(BatchNormalization(axis=1,input_shape=(10, 640,1)))
        # depth_i_model = Sequential()
        # depth_i_model.add(depth_model)
        depth_model.add(AveragePooling2D((2, 1), strides=(2, 1)))
        # depth_i_model.add(BatchNormalization())
        #
        # depth_model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu',dim_ordering = 'tf'))
        # depth_model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu',dim_ordering = 'tf'))
        # depth_model.add(AveragePooling2D((2, 2), strides=(2, 2)))
        # depth_model.add(BatchNormalization())
        # depth_model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu',dim_ordering = 'tf'))
        # # depth_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        # # depth_model.add(BatchNormalization())
        # # depth_model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu',dim_ordering = 'tf'))
        # depth_model.add(Convolution2D(320, 3, 3, border_mode='same', activation='relu',dim_ordering = 'tf'))
        # depth_model.add(AveragePooling2D((5, 64), strides=(5, 64)))
        #
        # depth_model.add(Reshape((5,320,1)))
        # depth_model.add(BatchNormalization())
        depth_model.add(Flatten())
        # merge_model = Sequential()
        # merge_model.add(Merge([depth_model, depth_i_model], mode='sum', concat_axis=1))
        # # merge_model.add(BatchNormalization())
        # merge_model.add(Flatten())
        return depth_model

    def create_conv_model(self):
        model = Sequential()
        model.add(BatchNormalization(axis=1,input_shape=(10, 640,3)))
        model.add(Convolution2D(32, 3, 3, border_mode='same', dim_ordering = 'tf',activation='relu'))
        # model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering = 'tf'))
        # model.add(BatchNormalization())
        model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering = 'tf' ))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering = 'tf' ))
        # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        # model.add(BatchNormalization())
        model.add(Convolution2D(128, 1, 1, border_mode='same', activation='relu', dim_ordering = 'tf' ))
        model.add(MaxPooling2D((4, 4), strides=(4, 4)))
        model.add(BatchNormalization())
        # model.add(MaxPooling2D((1, 4), strides=(1, 4)))
        # model.add(BatchNormalization())
        # model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering = 'tf' ))
        # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        # model.add(BatchNormalization())
        model.add(Flatten())
        return model

    def create_final_model(self, depth_model, model):
        final_model = Sequential()
        merge = Merge([depth_model, model],mode='concat')
        final_model.add(merge)
        final_model.add(BatchNormalization())
        final_model.add(Dense(1024, activation='relu'))
        final_model.add(Dense(1024, activation='relu'))
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
        self.final_model.compile(loss='mape', optimizer = self.sgd)

if __name__ == "__main__":
    pass
