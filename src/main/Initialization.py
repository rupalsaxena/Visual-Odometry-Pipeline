import cv2
import numpy as np
from scipy.spatial.distance import cdist

from Image import Image

from helpers import helpers


class Initialization:
    def __init__(self, img1, img2, K):
        self.image1 = img1
        self.image2 = img2
        self.K = K
        self.helpers = helpers()

    def run(self):
        img_obj1 = Image(self.image1)
        img_obj2 = Image(self.image2)

        keypoints1 = img_obj1.get_keypoints()
        keypoints2 = img_obj2.get_keypoints()

        keypoint_des1 = img_obj1.get_keypoints_descriptions()
        keypoint_des2 = img_obj2.get_keypoints_descriptions()
        
        keypoints_correspondence = self.get_keypoints_correspondence(keypoints1, keypoints2, keypoint_des1, keypoint_des2)

        E = self.getEssentialMatrix(keypoints_correspondence)
        R1, R2, T = cv2.decomposeEssentialMat(E)
        print(T)

    def get_keypoints_correspondence(self, keypoints1, keypoints2, keypoint_des1, keypoint_des2):
        """
        out: return corresponding matched keypoints between both images [[keypoints 1], [keypoints 2]]
        """
        matches = self.match_descriptors(keypoint_des1, keypoint_des2)

        kpt_matching = [[],[]]
        for idx, match in enumerate(matches):
            if(match is None):
                continue
            kpt1 = keypoints1[idx]
            kpt2 = keypoints2[match]

            kpt_matching[0].append(kpt1)
            kpt_matching[1].append(kpt2)

        return kpt_matching

    def match_descriptors(self, keypoint_des1, keypoint_des2):
        """
        Match keypoint descriptors using euclidean distance
        out: Matching list where matching[i] means keypoint_1[i] matches to keypoint_2[matching[i]]
        """
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
    
    def getEssentialMatrix(self, kpt_matching):
        """
        kpt_matching: keypoints matching between 2 images [[Point2D keypoint 1, Point2D keypoint 2]]
        """
        kpts1 = self.helpers.Point2DListToInt(kpt_matching[0])
        kpts2 = self.helpers.Point2DListToInt(kpt_matching[1])

        F, mask = cv2.findFundamentalMat(kpts1, kpts2, cv2.FM_RANSAC)

        return self.K.T @ F @ self.K
