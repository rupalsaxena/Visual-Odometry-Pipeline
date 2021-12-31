import cv2
import numpy as np
from helpers import helpers
import matplotlib.pyplot as plt 

class Continuous:
    def __init__(self, keypoints, landmarks, R, T, images, K):
        self.h = helpers()
        #self.init_R = R
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
        T_X = [self.init_T[0][0]]
        T_Y = [self.init_T[1][0]]

        p0 = self.h.Point2DListToInt(self.init_keypoints)
        p0 = np.float32(p0.reshape(-1, 1, 2))
        good_img_landmarks1 = self.init_landmarks

        for i in range(0, len(self.images)):
            # i less than equal to 2 is hard coded at the moment. If there is any change in 
            # choosing image frames for initialization, this has to change as well.
            if i<=2:
                continue
            
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.images[i-1], self.images[i], p0, None, **self.lk_params)

            if p1 is not None:
                good_img_keypoints2 = p1[st1==1]
                temp_lst= []

                for index, value in enumerate(st1):
                    if(value==1):
                        temp_lst.append(good_img_landmarks1[index])
                    
                good_img_landmarks1 = np.array(temp_lst)

            if len(good_img_keypoints2) > 4:
                retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                    good_img_landmarks1, good_img_keypoints2, self.K, None, flags=cv2.SOLVEPNP_P3P, confidence=0.9999
                )

                inliers = np.squeeze(np.array(inliers))
                good_img_keypoints2 = good_img_keypoints2[inliers,:]
                good_img_landmarks1 = good_img_landmarks1[inliers,:]

                T_X.append(tvec[0])
                T_Y.append(tvec[1])

                R, _ = cv2.Rodrigues(rvec)

                # kpts_obj = self.h.kpts2kpts2Object(good_img_keypoints2)
                # output_image = cv2.drawKeypoints(cv2.cvtColor(self.images[i], cv2.COLOR_GRAY2BGR), kpts_obj, 0, (0,255,0))

                # cv2.imshow('out', output_image)
                # cv2.waitKey(100)

            p0 = good_img_keypoints2.reshape(-1,1,2)

            # logic to add candidate keypoints
            
            # params for shiTomasi corner detection
            feature_params = dict(maxCorners = 1000,
                            qualityLevel = 0.01,
                            minDistance = 7,
                            blockSize = 7 )

            img_kpts = cv2.goodFeaturesToTrack(self.images[i], mask = None, **feature_params)
            img_kpts = np.squeeze(img_kpts)

            # img_kpts which are far away from good_img_keypoints2 are possible new_candidate kpts
            k = 0
            for idx in range(img_kpts.shape[0]):
                a = img_kpts[idx,:]
                min_norm = 1000000
                for j in range(good_img_keypoints2.shape[0]):
 
                    b = good_img_keypoints2[j,:]
                    norm = np.linalg.norm(a-b)
                    if min_norm>norm:
                        min_norm = norm

                # this norm can be tuned
                if min_norm > 10:
                    if k == 0:
                        new_candidate = a
                    else:
                        new_candidate = np.vstack([new_candidate,a])
                    k = k+1
            
            # i == 3 is hard coded at the moment. If there is any change in 
            # choosing image frames for initialization, this has to change as well.
            if i == 3:
                candidate_kpts = new_candidate
                rvec_candidate = np.zeros([new_candidate.shape[0],3])
                for j in range(new_candidate.shape[0]):
                    rvec_candidate[j,:] = rvec.T
                
            else:
                good_candidate_kpts, st, _ = cv2.calcOpticalFlowPyrLK(self.images[i-1], self.images[i], candidate_kpts, None, **self.lk_params)

                if good_candidate_kpts is not None:
                    good_candidate_kpts = good_candidate_kpts[st==1]

                    temp_lst= []
                    for index, value in enumerate(st):
                        if(value==1):
                            temp_lst.append(rvec_candidate[index])
                    rvec_candidate = np.array(temp_lst)
                            
                "select new_candidate that is far from candidate_kpts"
                k = 0
                for idx in range(new_candidate.shape[0]):
                    a = new_candidate[idx,:]
                    min_norm = 10000000
                    for j in range(good_candidate_kpts.shape[0]):
 
                        b = good_candidate_kpts[j,:]
                        norm = np.linalg.norm(a-b)
                        if min_norm>norm:
                            min_norm = norm

                    if min_norm > 10:
                        if k == 0:
                            selected_new_candidate = a
                        else:
                            selected_new_candidate = np.vstack([selected_new_candidate,a])
                        k = k+1
    
                "append selected new candidate to a final candidate array"
                candidate = np.vstack([good_candidate_kpts,selected_new_candidate])
            
                new_candidate_rvec = np.zeros([selected_new_candidate.shape[0],3])
                for l in range(selected_new_candidate.shape[0]):
                    new_candidate_rvec[l,:] = rvec.T
            
                rvec_candidate = np.vstack([rvec_candidate,new_candidate_rvec])
                candidate_kpts = candidate

            # plot candidate keypoints 
            candidate_kpts_obj = self.h.kpts2kpts2Object(candidate_kpts)
            output_image1 = cv2.drawKeypoints(cv2.cvtColor(self.images[i], cv2.COLOR_GRAY2BGR), candidate_kpts_obj, 0, (0,255,255))

            good_img_kpts_obj = self.h.kpts2kpts2Object(good_img_keypoints2)
            output_image2 = cv2.drawKeypoints(cv2.cvtColor(self.images[i], cv2.COLOR_GRAY2BGR), good_img_kpts_obj, 0, (0,255,0))

            horizontal_concat = np.concatenate((output_image1, output_image2), axis=1)
            cv2.imshow('left_candidate right_initialized', horizontal_concat)
            cv2.waitKey(100)
            candidate_kpts = candidate_kpts.reshape(-1,1,2)

        self.h.generate_trajectory(list(zip(T_X, T_Y)))