# import the necessary packages
from passion.gala.models import ResNet18
from keras.applications import ResNet50

# create the model dispatcher
MODEL_DISPATCHER = {
    "resnet50": ResNet50(weights = "imagenet", include_top = False,
                         input_shape = (120, 80, 3))
}
