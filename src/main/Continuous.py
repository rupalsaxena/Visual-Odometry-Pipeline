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
        self.lk_params = dict( winSize  = (49,49),
                  maxLevel = 7,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    def run(self):
        T_X = [self.init_T[2][0]]
        T_Y = [self.init_T[1][0]]
        # T_Y = []
        p0 = self.h.Point2DListToInt(self.init_keypoints)
        p0 = np.float32(p0.reshape(-1, 1, 2))
        good_img_landmarks1 = self.init_landmarks

        for i in range(0, len(self.images)):
            if i<=2:
                continue
            
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.images[i-1], self.images[i], p0, None, **self.lk_params)

            # print(sum(st))
            if p1 is not None:
                good_img_keypoints2 = p1[st==1]
                print(len(good_img_keypoints2))
                #good_img_keypoints1 = p0[st==1]
                temp_lst= []

                for index, value in enumerate(st):
                    if(value==1):
                        temp_lst.append(good_img_landmarks1[index])
                good_img_landmarks1 = np.array(temp_lst)
                # good_img_landmarks1 = good_img_landmarks1.reshape(-1,2)
                # good_img_landmarks1 = good_img_landmarks1[st==1]
            print(len(good_img_keypoints2), len(good_img_landmarks1))
            if len(good_img_keypoints2) > 20:
                kpts_obj = self.h.kpts2kpts2Object(good_img_keypoints2)
                output_image = cv2.drawKeypoints(cv2.cvtColor(self.images[i], cv2.COLOR_GRAY2BGR), kpts_obj, 0, (0,255,0))
                cv2.imshow('out', output_image)
                cv2.waitKey(100)
                retval, rvec, tvec, inliers = cv2.solvePnPRansac(good_img_landmarks1, good_img_keypoints2, self.K, None, flags=cv2.SOLVEPNP_P3P, confidence=0.99)
                T_X.append(tvec[0])
                T_Y.append(tvec[1])
                #R, _ = cv2.Rodrigues(rvec)

            p0 = good_img_keypoints2.reshape(-1,1,2)
        
        self.h.generate_trajectory(list(zip(T_X, T_Y)))
        
        # T_Y = [i for i in range(len(T_X))]
        # T_X,T_Y=T_Y, T_X
        # print([x for x in zip(T_X, T_Y)])

        

