# import the necessary packages
from passion.gala.models import ResNet18

# create the model dispatcher
MODEL_DISPATCHER = {
    "resnet18": ResNet18.build(32, 32, 3, 4)
}
