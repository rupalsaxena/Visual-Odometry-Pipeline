import cv2
import numpy as np
#from numpy.lib.type_check import imag
import matplotlib.pyplot as plt
from Image import Image
from helpers import helpers
import matplotlib.pyplot as plt 
#from scipy.spatial.distance import cdist
#from collections import deque

class Continuous:
    def __init__(self, keypoints, landmarks, R, T, images, K):
        self.h = helpers()
        self.init_R = R
        self.init_T = T
        self.K = K
        self.init_keypoints = keypoints
        self.init_landmarks = landmarks
        self.images = images
        self.init_keypoint_des = self.h.describe_keypoints(8, self.images[2], self.init_keypoints)
        self.images = list(map(np.uint8, self.images))
        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    def run(self):
        #img_des1 = self.init_keypoint_des
        T_X = []
        T_Y = []
        img_keypoints1 = self.init_keypoints
        img_keypoints1 = self.h.Point2DListToInt(img_keypoints1)
        p0 = np.float32(img_keypoints1.reshape(-1, 1, 2))

        #img_landmarks1 = self.init_landmarks
        good_img_landmarks1 = self.init_landmarks

        for i in range(0, len(self.images)):
            if i<=2:
                continue
            
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.images[i-1], self.images[i], p0, None, **self.lk_params)

            if p1 is not None:
                good_img_keypoints2 = p1[st==1]
                #good_img_keypoints1 = p0[st==1]
                temp_lst= []

                for index, value in enumerate(st):
                    if(value==1):
                        temp_lst.append(good_img_landmarks1[index])
                good_img_landmarks1 = np.array(temp_lst)
    
            if len(good_img_keypoints2) > 4:
                retval, rvec, tvec, inliers = cv2.solvePnPRansac(good_img_landmarks1, good_img_keypoints2, self.K, None)
                T_X.append(tvec[0])
                T_Y.append(tvec[1])
                #R, _ = cv2.Rodrigues(rvec)
            

            p0 = good_img_keypoints2.reshape(-1,1,2)
        
        print([x for x in zip(T_X, T_Y)])

        plt.plot(T_X, T_Y)
        plt.show()

        """for image_index, image in enumerate(self.images):

            img_landmarks2 = []
            img_obj = Image(image)

            img_keypoints2 = img_obj.get_keypoints()
            img_des2 = img_obj.get_keypoints_descriptions()

            kpts_correspondence = self.get_keypoints_correspondence(img_keypoints1, img_keypoints2, img_des1, img_des2)
            self.match_history.append(kpts_correspondence)
            import pdb; pdb.set_trace()
            if image_index == 0:
                for i, keypoint in enumerate(img_keypoints1):
                    if keypoint in kpts_correspondence[0]:
                        img_landmarks2.append(img_landmarks1[i])
            
            img_keypoints1 = img_keypoints2
            img_des1 = img_des2
            img_landmarks1 = img_landmarks2"""

            #print(len(kpts_correspondence[1]))
            #print(len(img_keypoints1), len(img_landmarks1))
            
    # def get_keypoints_correspondence(self, keypoints1, keypoints2, keypoint_des1, keypoint_des2):
    #     """
    #     out: return corresponding matched keypoints between both images [[keypoints 1], [keypoints 2]]
    #     """
    #     matches = self.match_descriptors(keypoint_des1, keypoint_des2)

    #     kpt_matching = [[],[]]
    #     for idx, match in enumerate(matches):
    #         if(match is None):
    #             continue
    #         kpt1 = keypoints1[idx]
    #         kpt2 = keypoints2[match]

    #         kpt_matching[0].append(kpt1)
    #         kpt_matching[1].append(kpt2)

    #     return kpt_matching

    # def match_descriptors(self, keypoint_des1, keypoint_des2):
    #     """
    #     Match keypoint descriptors using euclidean distance
    #     out: Matching list where matching[i] means keypoint_1[i] matches to keypoint_2[matching[i]]
    #     """
    #     MAX_DIST = 1e2
    #     done_des2 = set()
    #     matching = [None]*len(keypoint_des1)
    #     dist = cdist( keypoint_des1, keypoint_des2, metric="euclidean")

    #     for idx, dist_1 in enumerate(dist):
    #         while(True):
    #             min_match = np.argmin(dist_1)
    #             if(dist_1[min_match]==MAX_DIST):
    #                 break
    #             elif(min_match not in done_des2):
    #                 done_des2.add(min_match)
    #                 matching[idx] = min_match
    #                 break
    #             dist_1[min_match] = MAX_DIST

    #     return matching