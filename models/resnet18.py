# import the necessary packages
from classification_models.keras import Classifiers
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Dense, Activation
from keras.models import Model
from keras import backend as K


class ResNet18:
    @staticmethod
    def build(height, width, depth, classes):
        if K.image_data_format() == "channels_last":
            input_shape = (height, width, depth)
        else:
            input_shape = (depth, height, width)

        # initialize the base model
        resnet18, _ = Classifiers.get("resnet18")
        base_model = resnet18(input_shape = input_shape,
                              weights = "imagenet", include_top = False)

        # average pooling
        gap = GlobalAveragePooling2D()(base_model.output)

        # softmax classifier
        x = Dense(classes, kernel_initializer = "he_normal")(gap)
        x = Activation("softmax")(x)

        # return the constructed model architecture
        return Model(inputs = base_model.input, outputs = x)
