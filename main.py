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
from model import *


def main():
    vi = visualization()
    dh = data_handler()

    seed = 7
    np.random.seed(seed)

    folderLocation = os.path.dirname(os.path.realpath(__file__))
    DATA_HOME_DIR = folderLocation+''
    train_path = DATA_HOME_DIR+'train/'
    valid_path = DATA_HOME_DIR+'valid/'

    laser, rgb, depth, ret = dh.load(data_path = 'data/sensor/')

    if (ret==False):
        print "Problem with dataset"
        return
    N = 101
    N_plot = 100
    laser = laser[:N,:]
    rgb = rgb[:N,:,:,:]
    depth = depth[:N,:,:]

    km = keras_model(output_shape = laser.shape[1])

    print 'Laser: ' + str(laser.shape)
    print 'Depth: ' + str(depth.shape)
    print 'RGB: ' + str(rgb.shape)
    N = laser.shape[0]
    rgb_train = rgb

    directory = train_path+'rgb/'

    batch_size = 64
    no_of_epochs = 10
    km.compile()

    # km.final_model.load_weights('models/model_weights_200epoch.h5')
    km.final_model.fit([depth, rgb], laser, validation_split = 0.2,nb_epoch=no_of_epochs, batch_size=batch_size,callbacks=[km.tbCallback, km.lrate])
    km.final_model.save_weights('models/model_weights_200epoch.h5')

    # serialize model to JSON
    model_json = km.final_model.to_json()
    with open("models/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    km.final_model.save_weights("models/model.h5")
    print("Saved model to disk")
    # 
    # result = []
    # for  i in range(4)
    #     N  = N_plot + i*10
    #     result.append( km.final_model.predict([depth[N:N+1,:,:,:], rgb[N:N+1,:,:,:]]))


    # pose = [0.,0.,0.]
    # result_filter = pd.rolling_mean(result[0], 1)
    #
    # vi.depth_plot(np.mean(depth[N,:,:,:],axis=0),pose)
    # vi.laser_plot(laser[N,:],pose)
    # vi.laser_plot2(result_filter,pose)
    # plt.plot(0,0,'ok')
    # plt.show()
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(221)
    # ax1.plot([1,2,3,4,5], [10,5,10,5,10], 'r-')
    #
    # ax2 = fig.add_subplot(222)
    # ax2.plot([1,2,3,4], [1,4,9,16], 'k-')
    #
    # ax3 = fig.add_subplot(223)
    # ax3.plot([1,2,3,4], [1,10,100,1000], 'b-')
    #
    # ax4 = fig.add_subplot(224)
    # ax4.plot([1,2,3,4], [0,0,1,1], 'g-')




    # Data directory organization and import all the data for training

    # Define the model which will be trained
    # Train the model
    # Generate predictions
    # Validate predictions
def vgg_preprocess(x):
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr
def ConvBlock(model, layers, filters):
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(filters, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

if __name__ == "__main__":
    main()
    # predict()
