# import the necessary packages
from passion.gala.utils import MODEL_DISPATCHER
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications import imagenet_utils
from imutils import paths
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import pickle
import h5py
import os

# construct the argument parser to parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
                help = "path to input dataset")
ap.add_argument("-m", "--model", required = True,
                help = "path to model directory")
ap.add_argument("-db", "--database", required = True,
                help = "path HDF5 database")
ap.add_argument("-b", "--batch_size", type = int, default = 32,
                help = "batch size of images to be passed through network")
args = vars(ap.parse_args())

# initialize the database for reading
db = h5py.File(args["database"], "r")

# convenience variable
bs = 32

# grab the image paths and extract the image names from the paths
image_paths = list(paths.list_images(args["dataset"]))
image_names = [path.split(os.path.sep)[-1] for path in image_paths]

# initialize the feature extractor
cnn = MODEL_DISPATCHER["resnet50"]

# initialize a prediction dictionary
output = {"Image": [], "Class": []}

# loop through the image paths in batches
for i in tqdm(np.arange(0, len(image_paths), bs), total = np.ceil(len(image_paths) / bs)):
    # extract the current batch
    batch_paths = image_paths[i: i + bs]
    batch_names = image_names[i: i + bs]

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
    features = cnn.predict(batch_images, batch_size = bs)

    # flatten the feature vector
    features = features.reshape((features.shape[0], -1))

    # loop through the classifier folds
    predictions = []
    for fold in range(3):
        # build the path to the current classifier
        classifier_path = os.path.sep.join(
            [args["model"], f"resnet50_fold{fold}.cpickle"])

        # load the current model
        classifier = pickle.load(open(classifier_path, "rb"))

        # get the predictions of the current classifier
        preds = classifier.predict(features)

        # add the preds to the predictions list
        predictions.append(preds)

    # convert the predictions list to a numpy array
    predictions = np.array(predictions)

    # average the predictions
    predictions = list(np.round(np.mean(predictions, axis = 0), 0))

    # decode the predictions
    predictions = [db["label_names"][p] for p in predictions]

    # add the predictions to the output
    output["Image"].extend(batch_names)
    output["Class"].extend(predictions)

# create a submission dataframe
df = pd.DataFrame(output, columns = ["Image", "Class"])

# save the submission to disk
df.to_csv("../output/submission.csv", index = False)
