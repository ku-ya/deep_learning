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
from model import visualization
from model import data_handler
from model import keras_model
def main():
    vi = visualization()
    dh = data_handler()

    seed = 7
    np.random.seed(seed)

    folderLocation = os.path.dirname(os.path.realpath(__file__))
    DATA_HOME_DIR = folderLocation+'/data/sensor_data/'
    train_path = DATA_HOME_DIR+'train/'
    valid_path = DATA_HOME_DIR+'valid/'

    pose, laser, rgb, depth, ret = dh.load(data_path = 'data/sensor_pos_data/')

    if (ret==False):
        print "Problem with dataset"
        return

    km = keras_model(output_shape = laser.shape[1])
    print 'pose size: ' + str(pose.shape)

    cut_off = laser.shape[0]
    depth_s = depth[:cut_off,:,:,:]
    rgb_s = rgb[:cut_off,:,:,:]
    laser_s = laser[:cut_off,:]
    pose_s = pose[:cut_off,:]

    # train_datagen = image.ImageDataGenerator(
    #     rescale=1./255,
    #     shear_range=0,
    #     zoom_range=0,
    #     horizontal_flip=False)
    #
    # rgb_train = train_datagen.flow_from_directory(
    #     train_path+'rgb', batch_size=batch_size, target_size=(640,1))
    N = pose.shape[0]
    rgb_train = rgb

    directory = train_path+'rgb/'

    batch_size = 64
    no_of_epochs = 20


    laser_array = np.empty([N*64, 10])
    km.compile()

    # final_model.load_weights('models/model_weights_1.h5')
    # rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    print [depth_s, rgb_s, pose_s].shape
    km.final_model.fit([depth_s, rgb_s, pose_s], laser_s, validation_split = 0.3,nb_epoch=no_of_epochs, batch_size=batch_size,callbacks=[km.tbCallback, km.lrate])
    km.final_model.save_weights('models/model_weights_200epoch.h5')

    # pdb.set_trace()

    # serialize model to JSON
    model_json = km.final_model.to_json()
    with open("models/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    km.final_model.save_weights("models/model.h5")
    print("Saved model to disk")
    # batch_size = 64
    # no_of_epochs = 5
    #
    # model.fit(rgbd_train, laser_array, nb_epoch=no_of_epochs, batch_size=batch_size)
    #
    # no_of_epochs = 20
    # hl, = plt.plot([], [])

    # directory = valid_path+'depth/'
    # rgb_directory = valid_path+'rgb/'
    # laser_directory = valid_path+'laser_fov/'
    # valid_array = np.empty([1, 4, 640, 1])
    # for filename in os.listdir(directory):
    #     if filename.endswith(".jpg"):
    #         valid_array[0,0:3,:,0] = Image.open(rgb_directory+filename).load()
    #
    #     if filename.endswith(".npy"):
    #         # print(os.path.join(directory, filename))
    #         # print(filename)
    #         actual = np.load(laser_directory+filename).reshape(1,640)
    #         valid_array[0,3,:,0] = np.load(directory+filename).reshape(1,640)
# def predict():
#     filename = 'models/model.json'
#     with open(filename,'r') as f:
#         data = json.loads(f.read())
#         model = model_from_json(data)
    # pdb.set_trace()
    N = 750
    tic = time.clock()
    result = km.final_model.predict([depth[N:N+1,:,:,:], rgb[N:N+1,:,:,:], pose[N:N+1,:,:]])
    print time.clock() - tic

    #         print("prediction!")
    #         print(result.shape)
    #         plt.plot(actual[0],'ro')
    #         plt.plot(result[0],'x')
    pose = [0.,0.,0.]

    result_filter = pd.rolling_mean(result[0], 10)
    vi.depth_plot(np.mean(depth[N,:,:,:],axis=0),pose)
    vi.laser_plot(laser[N,:],pose)
    vi.laser_plot2(result_filter,pose)
    plt.plot(0,0,'ok')
    plt.show()
    #         time.sleep(0.2)
    #         break
    #
    #         continue
    #     else:
    #         continue
    #
    #
    #
    # return
    # plt.plot(x, np.transpose(x_train),'o')
    #
    # y_pred = model.predict(x.reshape(1,640))
    #
    # plt.plot(x, np.transpose(y_pred))
    # plt.show()


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
