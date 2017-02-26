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
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages

class visualization(object):
  """docstring for visualization."""
  # fig = plt.figure(num=None, figsize=(3.45, 2.5), dpi=80)

  def __init__(self):
    super(visualization, self).__init__()
    # self.fig = plt.figure(num=None, figsize=(3.45, 2.5), dpi=80)
    # self.ax = self.fig.add_subplot(111)
    # self.fov((0.,0.), -30, 30,100,fill=True, color='blue')

  def start_fig(self, dim):
    self.fig = plt.figure(num=None, figsize=dim, dpi=80)
    self.ax = self.fig.add_subplot(1, 1, 1)
    self.fov((0.,0.), -30, 30,100,fill=True, color='blue')
    self.robot()

  def set_lim(self, xlim,ylim, lloc, legend_flag):
    # ax = self.fig.gca()
    # self.ax.set_xticks(np.arange(-1,10,1))
    # self.ax.set_yticks(np.arange(-2,3,1))
    plt.xlabel('x [m]',fontsize=10)
    plt.ylabel('y [m]',fontsize=10)
    if legend_flag:
      plt.legend(loc=lloc,fontsize=8)
    plt.grid()
    plt.axis('equal')
    self.ax.set_xlim(xlim)
    self.ax.set_ylim(ylim)

  def save_fig(self, filename):
    plt.tight_layout()
    pp = PdfPages(filename)
    pp.savefig(self.fig)
    pp.close()

  def robot(self):
    self.ax.plot(0,0,'or',markersize=10)
    self.ax.plot([0, 0.2],[0,0],'k',lw=2)

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

  def laser_plot2(self, data, N_in, style, label_in):
    r = data
    N = N_in
    resol = 0.25
    resol = 58./N
    laser_angle_range = resol*N/180*np.pi
    # angle_offset = -18./180*np.pi
    laser_angle = np.linspace(-1./2*laser_angle_range, 1./2*laser_angle_range, N, endpoint=True)
    plt.plot(r*np.cos(laser_angle) , r*np.sin(laser_angle) , style, label=label_in)
    # self.set_lim()

  def fov(self, center,theta1, theta2, resolution, **kwargs):
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    radius = 0.8
    points = np.vstack((radius*np.cos(theta) + center[0],
                        radius*np.sin(theta) + center[1]))
    radius = 3.5
    theta = theta[::-1]
    point_outer = np.vstack((radius*np.cos(theta) + center[0],
                        radius*np.sin(theta) + center[1]))
    points = np.concatenate((points, point_outer), axis=1)
    # build the polygon and add it to the axes
    poly = patches.Polygon(points.T, closed=True, alpha=0.2, **kwargs)
    self.ax.add_patch(poly)

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
