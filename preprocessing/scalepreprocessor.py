# import the necessary packages
import numpy as np


class ScalePreprocessor:
    def __init__(self):
        pass

    def process(self, img):
        return (img.astype(np.float32) * (255 / img.max()))
