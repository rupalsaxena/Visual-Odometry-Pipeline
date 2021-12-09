import os
import cv2
import numpy as np

from Point2D import Point2D
from Point3D import Point3D

class helpers:
    def __init__(self):
        pass
    
    def loadImages(self, img_dir):
        "return list of numpy array of images"
        imagesList = []
        valid_images = (".jpg",".gif",".png",".tga")
        
        index = 0
        for file in sorted(os.listdir(img_dir)):
            if file.endswith(valid_images) &(index <10):
                imagesList.append(file)    
                index +=1
                
        loadedImages = []
        for image in imagesList:
            img = cv2.imread(img_dir + image)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = np.float32(img)
            loadedImages.append(img)

        return loadedImages
    
    def load_poses(self, K_file):
        "Load poses K from K file"

        with open(K_file) as f:
            lines = f.readlines()     
            lines = [line.strip(", \n") for line in lines]
            K = np.genfromtxt(lines, dtype = float, delimiter = ", ")
        return K
        
    def Point2DListToInt(self, keypoints):
        """
        keypoints: List of Point2D objects
        out: List of [x,y] keypoints in integers
        """
        return np.array([[k.u,k.v] for k in keypoints])
    
    def IntListToPoint2D(self, keypoints):
        """
        keypoints: List of [x,y] keypoints in integers
        out: List of Point2D objects
        """
        return [Point2D(keypoints[0][idx], keypoints[1][idx]) for idx in range(0, len(keypoints[0]))]
    
    def IntListToPoint3D(self, landmarks):
         """
        keypoints: List of [X, Y, Z] landmarks in integers
        out: List of Point3D objects
        """
        return [Point3D(landmarks[0][idx], landmarks[1][idx], landmarks[2][idx]) for idx in range(0, len(landmarks[0]))]
    
    def describe_keypoints(self, r, image, keypoints):
        # out: return the List of keypoint descriptors
        size = 2*r + 1
        keypoints_decription = np.zeros((len(keypoints), size*size))
        temp_img = np.pad(image, [(r, r), (r,r)], mode='constant')
        
        for idx, kpt in enumerate(keypoints):
            patch = temp_img[kpt.u:kpt.u + size, kpt.v:kpt.v + size].flatten()
            keypoints_decription[idx,:] = patch

        keypoints_decription = np.array(keypoints_decription)
        return keypoints_decription