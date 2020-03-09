# import the necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import argparse
import pickle
import h5py

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required = True,
                help = "path HDF5 database")
ap.add_argument("-m", "--model", required = True,
                help = "path to output model")
ap.add_argument("-j", "--jobs", type = int, default = -1,
                help = "# of jobs to run when tuning hyperparameters")
args = vars(ap.parse_args())

# initialize the database for reading
db = h5py.File(args["db"], "r")

# load the csv file
df = pd.read_csv("../input/mod_train_folds.csv")

# initialize the folds
train_folds = [1, 2]
val_folds = [0]

# get the appropriate fold indices
train_idxs = list(np.where(df.kfold.isin(train_folds))[0])
val_idxs = list(np.where(df.kfold.isin(val_folds))[0])

# # define the set of hyperparameters
# print("[INFO] tuning hyperparameters...")
# params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
# model = GridSearchCV(LogisticRegression(max_iter = 1000), params,
#                      cv = 3, n_jobs = args["jobs"], verbose = 10)
# model.fit(db["features"][train_idxs], db["labels"][train_idxs])
# print(f"[INFO] best hyperparameters: {model.best_params_}")

# training the model
print("[INFO] training the model...")
model = LogisticRegression(
    C = 1.0, n_jobs = args["jobs"], verbose = 1, max_iter = 1000)
model.fit(db["features"][train_idxs], db["labels"][train_idxs])

# evaluate the model
print("[INFO] evaluating the model...")
preds = model.predict(db["features"][val_idxs])
print(classification_report(db["labels"][val_idxs],
                            preds, target_names = db["label_names"]))

# serialize the model to disk
f = open(args["model"], "wb+")
f.write(pickle.dumps(model))
f.close()

# close the database
db.close()
