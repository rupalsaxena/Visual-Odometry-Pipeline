import cv2
import numpy as np
from helpers import helpers
import matplotlib.pyplot as plt 

class Continuous:
    def __init__(self, keypoints, landmarks, R, T, images, K):
        self.h = helpers()
        self.init_R = R
        self.init_T = T
        self.K = K
        self.init_keypoints = keypoints
        self.init_landmarks = landmarks
        self.images = list(map(np.uint8, images))

        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    def run(self):
        T_X = []
        T_Y = []
        p0 = self.h.Point2DListToInt(self.init_keypoints)
        p0 = np.float32(p0.reshape(-1, 1, 2))
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
                retval, rvec, tvec, inliers = cv2.solvePnPRansac(good_img_landmarks1, good_img_keypoints2, self.K, None, flags=cv2.SOLVEPNP_P3P, confidence=0.9999)
                T_X.append(tvec[0])
                T_Y.append(tvec[1])
                #R, _ = cv2.Rodrigues(rvec)
            

            p0 = good_img_keypoints2.reshape(-1,1,2)
        
        print([x for x in zip(T_X, T_Y)])

        plt.plot(T_X, T_Y)
        plt.show()