import os, sys
current_dir = os.getcwd()
from utils import *
from keras.layers import Merge
import time
# import imageio
from keras.models import model_from_json
import bcolz



data_path = 'data/sensor_pos_data/'
tic = time.clock()
pose = load_array(data_path+'pose.dat')
laser = load_array(data_path+'laser.dat')
rgb = load_array(data_path+'rgb.dat')
depth = load_array(data_path+'depth.dat')
print 'time to load data: ' + str(time.clock() - tic)

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
def main():
    seed = 7
    np.random.seed(seed)

    fname_dataset = 'full_data.mat'
    fname_model = 'model_cnn'
    folderLocation = os.path.dirname(os.path.realpath(__file__))
    DATA_HOME_DIR = folderLocation+'/data/sensor_data/'
    train_path = DATA_HOME_DIR+'train/'
    valid_path = DATA_HOME_DIR+'valid/'

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
    i = 0
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            # print(os.path.join(directory, filename))
            # print(filename)
            image = imageio.imread(directory+filename).T

            for j in range(64):
                rgbd_train[i+j,:,:,:] = image[:,j*10:(j+1)*10,:]
            i = i + 64
            continue
        else:
            continue

    # print("Training rgb shape: " )
    depth_array = np.empty([N*64, 1, 10, 10])
    directory = train_path+'depth/'
    i = 0
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            # print(os.path.join(directory, filename))
            # print(filename)
            # rgbd_train[i,3,:] = np.load(directory+filename)
            image = np.load(directory+filename).T
            # print(image.shape)
            for j in range(64):
                depth_array[j+i,0,:,:] =  image[j*10:(j+1)*10,:]
            i = i + 64
            continue
        else:
            continue
    # print('Depth train size: '+str(depth_array.shape))
    print('depth train size: '+str(depth_array.shape))
    print('rgb train size: '+str(rgbd_train.shape))

    laser_array = np.empty([N*64, 10])
    i = 0
    directory = train_path+'laser_fov/'
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            # print(os.path.join(directory, filename))
            # print(filename)
            laser_array[i*64:(i+1)*64,:] = np.load(directory+filename).reshape([64,10])
            i = i + 1
            continue
        else:
            continue
    print('Laser train size: '+ str(laser_array.shape))


    depth_model = Sequential()
    depth_model.add(BatchNormalization(axis=1,input_shape=(1,10, 10)))
    depth_model.add(Dense(10, activation='relu'))
    depth_model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu',dim_ordering = 'th'))
    # depth_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    depth_model.add(BatchNormalization())
    # depth_model.add(MaxPooling1D(2,stride=2))
    # depth_model.add(Dropout(0.2))
    # depth_model.add(Convolution1D(nb_filter=64,
    #                               filter_length=3,
    #                               border_mode='same',
    #                               activation='relu'))
    # depth_model.add(MaxPooling1D(2,stride=2))
    # depth_model.add(Dropout(0.2))
    # depth_model.add(Convolution1D(nb_filter=128,
    #                               filter_length=3,
    #                               border_mode='same',
    #                               activation='relu'))
    # depth_model.add(MaxPooling1D(2,stride=2))
    # depth_model.add(BatchNormalization())
    # depth_model.add(Dense(640, input_dim=640, init='he_normal', activation='relu'))
    # depth_model.add(Convolution2D(32, 3, 1,border_mode='same',activation='relu'))
    depth_model.add(Flatten())
    # depth_model.add(Dense(1000,activation='relu'))
    # depth_model.add(BatchNormalization())
    # depth_model.add(Dropout(0.5))
    # depth_model.add(Dense(640,activation='linear'))

    # depth_model.summary()
    batch_size = 6400
    no_of_epochs = 10
    # depth_model.fit(depth_array, laser_array, nb_epoch=no_of_epochs, batch_size=batch_size)



    model = Sequential()
    model.add(BatchNormalization(axis=1,input_shape=(3, 10, 10)))
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 10, 10), dim_ordering = 'th',activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering = 'th'))
    # model.add(BatchNormalization())
    # model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', dim_ordering = 'th' ))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    # model.add(Convolution2D(64, 3, 1, border_mode='same',activation='relu'))
    # model.add(MaxPooling2D((1, 2), strides=(1, 2)))
    # model.add(BatchNormalization())
    # model.add(Convolution2D(128, 3, 1, border_mode='same',activation='relu'))
    # model.add(MaxPooling2D((1, 2), strides=(1, 2)))
    # model.add(BatchNormalization())
    #
    # model.add(Dropout(0.5))


    model.add(Flatten())


    # model.add(Dense(640,activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(640,activation='linear'))
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='mean_squared_error', optimizer=sgd)
    #
    final_model = Sequential()
    merge = Merge([depth_model, model],mode='concat')

    final_model.add(merge)
    final_model.add(Dense(1000, activation='linear'))
    final_model.add(Dense(10, activation='linear'))
    final_model.summary()
    # final_model.load_weights('models/model_weights_1.h5')
    sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    final_model.compile(loss='mae', optimizer=sgd)
    final_model.fit([depth_array, rgbd_train], laser_array, nb_epoch=no_of_epochs, batch_size=batch_size)
    #
    #
    final_model.save_weights('models/model_weights_1.h5')

    # serialize model to JSON
    model_json = final_model.to_json()
    with open("models/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    final_model.save_weights("models/model.h5")
    print("Saved model to disk")

    #

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
    #         result = model.predict(valid_array)
    #         print("prediction!")
    #         print(result.shape)
    #         plt.plot(actual[0],'ro')
    #         plt.plot(result[0],'x')
    #         plt.plot(valid_array[0,3,:,0],'go')
    #         plt.show()
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
