from random import triangular
import cv2
import numpy as np
from numpy.core.numeric import identity
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

        E, inliers1, inliers2 = self.getEssentialMatrix(keypoints_correspondence)
        R1, R2, T = cv2.decomposeEssentialMat(E)      

        self.disambiguateEssential(E, inliers1, inliers2)  


    def disambiguateEssential(self, E, inliers1, inliers2):
        R1, R2, T = cv2.decomposeEssentialMat(E) 
        # inliers1 = np.vstack((inliers1, np.ones(inliers1.shape[1])))
        # inliers2 = np.vstack((inliers2, np.ones(inliers1.shape[1])))


        # print(R1, R2)
        self.triangulate(R1, T, inliers1, inliers2)
        self.triangulate(R2, T, inliers1, inliers2)
                # print("\n\n")
        self.triangulate(R1, -T, inliers1, inliers2)
        self.triangulate(R2, -T, inliers1, inliers2)

    
    def triangulate(self, R,t, inliers1, inliers2):
        inliers1 = np.vstack((inliers1, np.ones(inliers1.shape[1])))
        inliers2 = np.vstack((inliers2, np.ones(inliers1.shape[1])))

        norm_inliers1 = np.linalg.inv(self.K) @ inliers1
        norm_inliers2 = np.linalg.inv(self.K) @ inliers2

        identity = cv2.hconcat([np.eye(3), np.zeros((3,1))])
        M1 = np.concatenate((R, t), axis=1) 

        points3D = cv2.triangulatePoints(identity, M1, norm_inliers1[:2,:], norm_inliers2[:2,:])
        points3D /= points3D[3]
        
        relative_p1 = identity @ points3D
        relative_p2 = M1 @  points3D

        return (points3D, sum(relative_p1[2]>0), sum(relative_p2[2]>0))


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

        inliers1 = np.array([[kpts1[i][0], kpts1[i][1]] for i,j in enumerate(mask) if j[0] == 1]).T
        inliers2 = np.array([[kpts2[i][0], kpts2[i][1]] for i,j in enumerate(mask) if j[0] == 1]).T
        return self.K.T @ F @ self.K, inliers1, inliers2
