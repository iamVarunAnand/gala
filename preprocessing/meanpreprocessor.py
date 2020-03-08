# import the necessary packages
import numpy as np


class MeanPreprocessor:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def process(self, img):
        return np.divide(np.subtract(img, self.mean), self.std).astype(np.float32)
