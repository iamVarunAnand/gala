# import the necessary packages
from keras.backend.tensorflow_backend import set_session
from passion.gala.preprocessing import SimplePreprocessor
from passion.gala.preprocessing import ScalePreprocessor
from passion.gala.preprocessing import MeanPreprocessor
from passion.gala.utils import MODEL_DISPATCHER
from passion.gala.utils import GalaDataGenerator
from keras.optimizers import Adam
import pandas as pd
import numpy as np
import json
import os

# tensorflow session settings
# config = tf.compat.v1.ConfigProto(allow_soft_placement = True)
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# set_session(sess)

# training params
EPOCHS = 10
BATCH_SIZE = 64

# load the csv file
df = pd.read_csv("../input/mod_train_folds.csv")

# read the dataset variables into a dictionary
f = open("../input/dataset_vars.json", "r")
dataset_vars = json.load(f)
f.close()

# initialize the preprocessors
sp = SimplePreprocessor(32, 32)
scp = ScalePreprocessor()
mp = MeanPreprocessor(dataset_vars["mean"], dataset_vars["std"])

# initialize the data generators
train_datagen = GalaDataGenerator(
    df, [0, 1], 32, 64, preprocessors = [sp])
val_datagen = GalaDataGenerator(df, [2], 32, BATCH_SIZE, preprocessors = [sp])

# initlalize the model
model = MODEL_DISPATCHER["resnet18"]

# initialize the optimizer
opt = Adam(lr = 3e-4)

# compile the model
model.compile(loss = "categorical_crossentropy",
              optimizer = opt, metrics = ["accuracy"])

# fit the model
model.fit_generator(train_datagen.generator(),
                    steps_per_epoch = np.ceil(
                        train_datagen.num_images / BATCH_SIZE),
                    epochs = EPOCHS,
                    validation_data = val_datagen.generator(),
                    validation_steps = np.ceil(
                        val_datagen.num_images / BATCH_SIZE),
                    max_queue_size = 2 * BATCH_SIZE)
