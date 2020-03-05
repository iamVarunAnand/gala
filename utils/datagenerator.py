# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from cv2 import cv2
import pandas as pd
import numpy as np
import os


class GalaDataGenerator:
    def __init__(self, df, folds, thresh, batch_size, preprocessors = None,
                 aug = None):
        # clean up the input dataframe
        df = df[df.kfold.isin(folds)]
        df = df[df.height > thresh]

        # iniialize the instance variables
        self.df = df[["Image", "Class"]]
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.aug = aug
        self.lb = LabelBinarizer().fit(df.Class)

    def generator(self, passes = np.inf):
        # define the base path to the input images
        BASE_PATH = "../input/train_images"

        # initialize a variable to store the number of epochs
        epochs = 0

        # loop indefinitely
        while(epochs < passes):
            # grab the list of possible indices and shuffle them
            idxs = list(range(self.df.shape[0]))
            np.random.shuffle(idxs)

            # loop through the indices in batches
            for i in range(0, len(idxs), self.batch_size):
                # sort the list of current indices
                cur_idxs = sorted(idxs[i: i + self.batch_size])

                # loop through the indices
                imgs = []
                lbls = []
                for idx in cur_idxs:
                    # build the path to the current image
                    img_path = os.path.sep.join([BASE_PATH, self.df.Image[idx]])

                    # read the image from disk
                    img = cv2.imread(img_path)

                    # if any preprocessors are supplied apply them
                    if self.preprocessors is not None:
                        # loop through the preprocessors
                        for p in self.preprocessors:
                            img = p.process(img)

                    # extract the appropriate label
                    lbl = self.df.Class[idx]

                    # add the image and labels to their lists
                    imgs.append(img)
                    lbls.append(lbl)

                # convert the images and labels into a numpy array
                imgs = np.array(imgs)
                lbls = np.array(lbls)

                # convert the labels from strings into vectors
                lbls = self.lb.transform(lbls)

                # if any augmentor is supplied, apply it
                if self.aug is not None:
                    (imgs, lbls) = self.aug.flow(
                        imgs, lbls, batch_size = self.batch_size)

                # yield the current batch
                yield (np.array(imgs), np.array(lbls))
