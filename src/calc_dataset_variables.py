# import the necessary packages
from tqdm import tqdm
from cv2 import cv2
import numpy as np
import glob
import json


def get_mean(arr):
    return np.mean(arr)


# grab the image paths
img_paths = glob.glob("../input/train_images/*.jpg")

# loop through the image paths
R, G, B = [], [], []
R2, G2, B2 = [], [], []

for img_path in tqdm(img_paths, total = len(img_paths)):
    # read the image and split it
    img = cv2.imread(img_path)
    b, g, r = cv2.split(img)

    # calculate the variables
    R.append(np.mean(r))
    G.append(np.mean(g))
    B.append(np.mean(b))
    R2.append(np.mean(np.multiply(r, r, dtype = np.float32)))
    G2.append(np.mean(np.multiply(g, g, dtype = np.float32)))
    B2.append(np.mean(np.multiply(b, b, dtype = np.float32)))

# calculate the per channel mean and std
rmean = get_mean(R)
rstd = np.sqrt(get_mean(R2) - np.multiply(rmean, rmean))

gmean = get_mean(G)
gstd = np.sqrt(get_mean(G2) - np.multiply(gmean, gmean))

bmean = get_mean(B)
bstd = np.sqrt(get_mean(B2) - np.multiply(bmean, bmean))

# create the dict
dataset_vars = {"mean": [rmean, gmean, bmean],
                "std": [rstd, gstd, bstd]}

# write the variables to a json file
f = open("../input/dataset_vars.json", "w+")
json.dump(dataset_vars, f, indent = 4)
f.close()
