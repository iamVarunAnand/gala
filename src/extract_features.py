# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from passion.gala.utils import MODEL_DISPATCHER
from passion.gala.utils import HDF5DatasetWriter
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications import imagenet_utils
from imutils import paths
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import os

# construct an argument parser to parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
                help = "path to input dataset")
ap.add_argument("-o", "--output", required = True,
                help = "path to output hdf5 file")
ap.add_argument("-b", "--batch_size", type = int, default = 32,
                help = "batch size of images to be passed through network")
args = vars(ap.parse_args())

# convenience variable
bs = args["batch_size"]

# read the csv file
df = pd.read_csv(os.path.sep.join([args["dataset"], "mod_train_folds.csv"]))

# build the image paths along with the labels
image_paths = [os.path.sep.join(
    [args["dataset"], "train_images", df.Image[i]]) for i in range(df.shape[0])]
labels = [df.Class[i] for i in range(df.shape[0])]

# convert the labels from integers into vectors
le = LabelEncoder()
labels = le.fit_transform(labels)

# load the model
model = MODEL_DISPATCHER["resnet50"]

# initialize the HDF5 dataset writer and store the class labels
db = HDF5DatasetWriter((len(image_paths), 4 * 3 * 2048),
                       args["output"], data_key = "features")
db.store_class_labels(le.classes_)

# loop over the image in batches
for i in tqdm(np.arange(0, len(image_paths), bs), total = np.ceil(len(image_paths) / bs)):
    # extract the current batch
    batch_paths = image_paths[i: i + bs]
    batch_labels = labels[i: i + bs]

    # loop over the images and labels in the current batch
    batch_images = []
    for (j, image_path) in enumerate(batch_paths):
        # load the image and convert it into keras compatible form
        img = load_img(image_path, target_size = (120, 80))
        img = img_to_array(img)

        # preprocess the image
        img = np.expand_dims(img, axis = 0)
        img = imagenet_utils.preprocess_input(img)

        # add the image to the batch
        batch_images.append(img)

    # pass the current batch through the model and obtain the predictions
    batch_images = np.vstack(batch_images)
    features = model.predict(batch_images, batch_size = bs)

    # flatten the feature vector
    features = features.reshape((features.shape[0], -1))

    # add the features and the labels to the dataset
    db.add(features, batch_labels)

# close the dataset
db.close()
