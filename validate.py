import os, sys
from utils import *
from keras.layers import Merge
import time
import imageio
from keras.models import model_from_json
def main():
    folderLocation = os.path.dirname(os.path.realpath(__file__))
    DATA_HOME_DIR = folderLocation+'/data/sensor_data/'
    train_path = DATA_HOME_DIR+'train/'
    valid_path = DATA_HOME_DIR+'train/'
    # model = Sequential()
    # load json and create model
    json_file = open('models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("models/model_weights_1.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='mean_squared_error', optimizer='Adagrad')
    # score = loaded_model.evaluate(X, Y, verbose=0)
    # print "%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100)

    directory = valid_path+'depth/'
    rgb_directory = valid_path+'rgb/'
    laser_directory = valid_path+'laser_fov/'


    valid_rgb_array = np.empty([1, 3, 640, 1])
    valid_depth_array = np.empty([1, 640, 1])
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            valid_rgb_array[0,:,:,0] = Image.open(rgb_directory+filename).load()

        if filename.endswith(".npy"):
            # print(os.path.join(directory, filename))
            # print(filename)
            actual = np.load(laser_directory+filename).reshape(1,640)
            valid_depth_array[0,:,0] = np.load(directory+filename).reshape(1,640)
            result = loaded_model.predict([valid_depth_array, valid_rgb_array])
            print("prediction!")
            print(result.shape)
            plt.plot(actual[0],'ro')
            plt.plot(result[0],'x')
            plt.plot(valid_depth_array[0,:,:],'go')
            plt.show()
            time.sleep(0.2)
            break

            continue
        else:
            continue



    return
    plt.plot(x, np.transpose(x_train),'o')

    y_pred = model.predict(x.reshape(1,640))

    plt.plot(x, np.transpose(y_pred))
    plt.show()

if __name__ == "__main__":
    main()
