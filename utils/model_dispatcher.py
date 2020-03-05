# import the necessary packages
from classification_models.keras import Classifiers

# create the model dispatcher
MODEL_DISPATCHER = {
    "resnet18": Classifiers.get("resnet18")[0](input_shape = (32, 32, 3),
                                               weights = "imagenet",
                                               include_top = False)
}
