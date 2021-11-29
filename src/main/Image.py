import cv2 
import numpy as np 
from Point2D import Point2D
class Image:
    def __init__(self, image):
        self.image = image

        self.generate_harris_score()
        self.harris_keypoints = None
        self.keypoints_description = None
    
    def generate_harris_score(self):
        # TODO: Tune parameters of cornerHarris funtion from opencv
        self.harris_score = cv2.cornerHarris(self.image,2,3,0.04)

    def select_keypoints(self, num, r):
        # input number of selected keypoints patch radius
        keypoints = [Point2D(0,0)]*num

