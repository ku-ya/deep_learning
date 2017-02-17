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


# print np.argwhere(np.isnan(depth))
# sys.exit()

# print 'time to load data: ' + str(time.clock() - tic)

# vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))

def depth_plot(data, pose):
    r = np.mean(data.T,axis=0)
    N = len(r)
    resol = 0.0906
    angle_range = resol*N/180*np.pi
    angle = np.linspace(-1./2*angle_range, 1./2*angle_range,N, endpoint=True) - pose[2]
    plt.plot(- pose[0] + r*np.cos(angle), - pose[1] + r*np.sin(angle),'g')

def laser_plot(data, pose):
    r = data
    N = len(data)
    resol = 0.25
    laser_angle_range = resol*N/180*np.pi
    angle_offset = -18./180*np.pi
    laser_angle = np.linspace(-1./2*laser_angle_range, 1./2*laser_angle_range, N, endpoint=True) - pose[2]
    plt.plot(r*np.cos(laser_angle) - pose[0], r*np.sin(laser_angle) - pose[1],'b')
def laser_plot2(data, pose):
    r = data
    N = len(data)
    resol = 0.25
    laser_angle_range = resol*N/180*np.pi
    angle_offset = -18./180*np.pi
    laser_angle = np.linspace(-1./2*laser_angle_range, 1./2*laser_angle_range, N, endpoint=True) - pose[2]
    plt.plot(r*np.cos(laser_angle) - pose[0], r*np.sin(laser_angle) - pose[1], 'r')

def main():
    seed = 7
    np.random.seed(seed)

    fname_dataset = 'full_data.mat'
    fname_model = 'model_cnn'
    folderLocation = os.path.dirname(os.path.realpath(__file__))
    DATA_HOME_DIR = folderLocation+'/data/sensor_data/'
    train_path = DATA_HOME_DIR+'train/'
    valid_path = DATA_HOME_DIR+'valid/'


    data_path = 'data/sensor_pos_data/'
    # tic = time.clock()
    pose = load_array(data_path+'pose.dat')
    laser = load_array(data_path+'laser.dat')
    rgb = load_array(data_path+'rgb.dat')
    depth = load_array(data_path+'depth.dat')
    depth = depth[..., None]
    pose = pose[...,None]

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

    pose_model = Sequential()
    # pose_model.add(BatchNormalization(axis=1,input_shape=(3,1)))
    pose_model.add(Dense(1, input_shape=(3,1), activation='linear'))
    pose_model.add(Flatten())

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

    final_model = Sequential()
    merge = Merge([depth_model, model, pose_model],mode='concat')
    final_model.add(merge)
    # final_model.add(BatchNormalization())
    # final_model.add(Dense(10000, activation='relu'))
    final_model.add(Dense(laser.shape[1], activation='linear'))
    final_model.summary()

    def step_decay(epoch):
    	initial_lrate = 0.1
    	drop = 0.5
    	epochs_drop = 10.0
    	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    	return lrate
    # final_model.load_weights('models/model_weights_1.h5')
    sgd = SGD(lr=0.0, decay=1e-4, momentum=0.9, nesterov=True)
    # rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


    def mean_squared_error_exp(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true)*tf.exp(-tf.abs(y_true)/(1.5**2)), axis=-1)
    final_model.compile(loss=mean_squared_error_exp, optimizer=sgd)

    tbCallback=keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

    lrate = LearningRateScheduler(step_decay)
    final_model.fit([depth_s, rgb_s, pose_s], laser_s, validation_split = 0.3,nb_epoch=no_of_epochs, batch_size=batch_size,callbacks=[tbCallback, lrate])
    #
    #
    final_model.save_weights('models/model_weights_200epoch.h5')

    # pdb.set_trace()

    # serialize model to JSON
    model_json = final_model.to_json()
    with open("models/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    final_model.save_weights("models/model.h5")
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
    result = final_model.predict([depth[N:N+1,:,:,:], rgb[N:N+1,:,:,:], pose[N:N+1,:,:]])
    print time.clock() - tic

    #         print("prediction!")
    #         print(result.shape)
    #         plt.plot(actual[0],'ro')
    #         plt.plot(result[0],'x')
    pose = [0.,0.,0.]

    result_filter = pd.rolling_mean(result[0], 10)
    depth_plot(np.mean(depth[N,:,:,:],axis=0),pose)
    laser_plot(laser[N,:],pose)
    laser_plot2(result_filter,pose)
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
