# import the necessary packages
from sklearn.model_selection import StratifiedKFold
import pandas as pd

if __name__ == "__main__":
    # read the csv file
    df = pd.read_csv("../input/mod_train.csv")

    # create a new column to store the fold values
    df["kfold"] = -1

    # shuffle the dataset
    df = df.sample(frac = 1).reset_index(drop = True)

    # initialize the stratifier
    skf = StratifiedKFold(n_splits = 3)

    # create the folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df.Class.values)):
        # update the fold value
        df.kfold[val_idx] = fold

    # sanity check
    print(df.kfold.value_counts())

    # save the csv file
    df.to_csv("../input/mod_train_folds.csv", index = False)
