# import the necessary packages
from cv2 import cv2


class SimplePreprocessor:
    def __init__(self, height, width, inter = cv2.INTER_LINEAR):
        # initialize the instance variables
        self.height = height
        self.width = width
        self.inter = inter

    def process(self, img):
        # return the resized image
        return cv2.resize(img, (self.width, self.height), interpolation = self.inter)
