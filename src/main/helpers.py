import os
import cv2
import numpy as np

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