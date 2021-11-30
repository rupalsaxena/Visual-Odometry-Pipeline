import cv2
import numpy as np
from scipy.spatial.distance import cdist

from Image import Image


class Initialization:
    def __init__(self, img1, img2, K, num_iter=1200):
        self.image1 = img1
        self.image2 = img2
        self.K = K
        self.num_iter = num_iter
        
    def generate_keypoints_correspondences(self):
        img_obj1 = Image(self.image1)
        img_obj2 = Image(self.image2)

        keypoints1 = img_obj1.get_keypoints()
        keypoints2 = img_obj2.get_keypoints()

        keypoint_des1 = img_obj1.get_keypoints_descriptions()
        keypoint_des2 = img_obj2.get_keypoints_descriptions()

        keypoints_match = self.match_keypoints(keypoints1, keypoints2, keypoint_des1, keypoint_des2)
    
    def match_keypoints(self, keypoints1, keypoints2, keypoint_des1, keypoint_des2):
        # Thought! Maybe match keypoints should be a class or a function in helpers function 
        # since it can be used during localization as well
        # TODO: implement this method
        matches = self.match_descriptors(keypoint_des1, keypoint_des2)

        kpt_matching = []
        for idx, match in enumerate(matches):
            if(match is None):
                continue
            kpt1 = keypoints1[idx]
            ktp2 = keypoints2[match]

            kpt_matching.append([kpt1, ktp2])
                
        return kpt_matching

    def match_descriptors(self, keypoint_des1, keypoint_des2):
        # Match keypoint descriptors using euclidean distance
        # out: Matching list where matching[i] means keypoint_1[i] matches to keypoint_2[matching[i]] 
        MAX_DIST = 1e2
        done_des2 = set()
        matching = [None]*len(keypoint_des1)
        dist = cdist( keypoint_des1, keypoint_des2, metric="euclidean")

        for idx, dist_1 in enumerate(dist):
            while(True):
                min_match = np.argmin(dist_1)
                if(dist_1[min_match]==MAX_DIST):
                    break
                elif(min_match not in done_des2):
                    done_des2.add(min_match)
                    matching[idx] = min_match
                    break
                dist_1[min_match] = MAX_DIST

        return matching
