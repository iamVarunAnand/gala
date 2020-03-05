# import the necessary packages
from tqdm import tqdm
from cv2 import cv2
import pandas as pd
import os

# initialize the base path
BASE_PATH = "../input/train_images"

# read the dataframe
df = pd.read_csv("../input/train.csv")

# create a new column to store the image height
df["height"] = -1

# initialize the threshold
thresh = 32

# loop through the image paths
for i in tqdm(range(df.shape[0]), total = df.shape[0]):
    # build the path to the image
    img_path = os.path.sep.join([BASE_PATH, df.Image[i]])

    # read the image
    img = cv2.imread(img_path)

    # modify the height value of the corresponding record
    df.height[i] = img.shape[0]

# save the modified dataset to disk
df.to_csv("../input/mod_train.csv", index = False)
