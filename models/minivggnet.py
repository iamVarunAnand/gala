# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras import backend as K

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model and check the dataFormat to initialize input
        # shape accordingly
        model = Sequential()
        if K.image_data_format() == "channels_last":
            input_shape = (height, width, depth)
            channel_dim = -1
        else:
            input_shape = (depth, height, width)
            channel_dim = 1

        # first CONV => RELU => CONV => RELU => POOl set
        model.add(Conv2D(32, (3, 3), input_shape = input_shape, padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = channel_dim))
        model.add(Conv2D(32, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = channel_dim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        # second CONV => RELU => CONV => RELU => POOL set
        model.add(Conv2D(64, (3, 3), input_shape = input_shape, padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = channel_dim))
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = channel_dim))
        model.add(MaxPooling2D(pool_size = (2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = channel_dim))
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
