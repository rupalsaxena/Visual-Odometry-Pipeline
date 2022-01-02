import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

from helpers import helpers


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
        T_Y = [self.init_T[2][0]]

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
                good_img_keypoints_temp = good_img_keypoints2
                good_img_keypoints2 = good_img_keypoints2[inliers,:]
                good_img_landmarks1 = good_img_landmarks1[inliers,:]

                
                # inverse tvec and rvec
                R,_ = cv2.Rodrigues(rvec)
                R = R.T
                rvec,_ = cv2.Rodrigues(R)             
                tvec = -R @ tvec

                T_X.append(tvec[0])
                T_Y.append(tvec[2])
            

                #R, _ = cv2.Rodrigues(rvec)
            # logic to add candidate keypoints
            
            # First: calculate new keypoints using ShiTomasi
            feature_params = dict(maxCorners = 1000,
                            qualityLevel = 0.01,
                            minDistance = 7,
                            blockSize = 7 )

            img_kpts = cv2.goodFeaturesToTrack(self.images[i], mask = None, **feature_params)
            img_kpts = np.squeeze(img_kpts)

            # Second: img_kpts which are far away from good_img_keypoints2 are possible candidates kpts
            k = 0
            
            new_candidate = np.zeros((0,2)).astype('float32')
            for idx in range(img_kpts.shape[0]):
                a = img_kpts[idx,:]
                min_norm = 1e6
                for j in range(good_img_keypoints_temp.shape[0]):
 
                    b = good_img_keypoints_temp[j,:]
                    norm = np.linalg.norm(a-b)
                    if min_norm>norm:
                        min_norm = norm

                # this norm can be tuned
                if min_norm > 30:
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
                tvec_candidate = np.zeros([new_candidate.shape[0],3])
                for j in range(new_candidate.shape[0]):
                    rvec_candidate[j,:] = rvec.T
                    tvec_candidate[j,:] = tvec.T

                fir_obs_C = candidate_kpts
                
            elif(candidate_kpts.shape[0]>0):
                # print(candidate_kpts)
                # candidate_kpts = candidate_kpts.astype('float32')
                # Third: candidate_kpts is good if it is tracked in next frame
                            
                good_candidate_kpts, st, _ = cv2.calcOpticalFlowPyrLK(
                    self.images[i-1], self.images[i], candidate_kpts, None, **self.lk_params
                )

                if good_candidate_kpts is not None:
                    good_candidate_kpts = good_candidate_kpts[st==1]

                    # get rvec_candidate and fir_obs_C based on succesful tracking of candidate kpts
                    temp_rvec= []
                    temp_tvec = []
                    temp_fir_obs_C = []

                    for index, value in enumerate(st):
                        if(value==1):
                            temp_rvec.append(rvec_candidate[index])
                            temp_tvec.append(tvec_candidate[index])
                            temp_fir_obs_C.append(fir_obs_C[index])


                    rvec_candidate = np.array(temp_rvec)
                    tvec_candidate = np.array(temp_tvec)
                    fir_obs_C = temp_fir_obs_C

                # Fourth: Find completely new candidate kpts which are never found is previous frames
                # So new_candidate kpts which are far from good_candidate_kpts are completely new candidate kpts.
                k = 0

                selected_new_candidate = np.zeros((0,2)).astype('float32')
                for idx in range(new_candidate.shape[0]):
                    if(len(new_candidate.shape)!=2):
                        break
                    print(new_candidate.shape)
                    a = new_candidate[idx,:]
                    min_norm = 10000000
                    for j in range(candidate_kpts.shape[0]):
 
                        b = candidate_kpts[j,:]
                        norm = np.linalg.norm(a-b)
                        if min_norm>norm:
                            min_norm = norm
                            
                    # this norm can be tuned too

                    if min_norm > 30:
                        if k == 0:
                            selected_new_candidate = a
                        else:
                            selected_new_candidate = np.vstack([selected_new_candidate,a])
                        k = k+1
                
                # getting rvec and tvec of completely new candidates keypoints
                new_candidate_rvec = np.zeros([selected_new_candidate.shape[0],3])
                new_candidate_tvec = np.zeros([selected_new_candidate.shape[0],3])
                for l in range(selected_new_candidate.shape[0]):
                    new_candidate_rvec[l,:] = rvec.T
                    new_candidate_tvec[l,:] = tvec.T

                #Fifth: stacking old with newly created keypoints
                candidate_kpts = np.vstack([good_candidate_kpts,selected_new_candidate]) # C as per problem statement
                fir_obs_C = np.vstack([fir_obs_C, selected_new_candidate]) # F as per problem statement
                rvec_candidate = np.vstack([rvec_candidate,new_candidate_rvec]) # T as per problem statement (but only rotation)
                tvec_candidate = np.vstack([tvec_candidate, new_candidate_tvec]) # T as per problem statement (but only translation)

                # Remove duplicate candidates
                candidate_kpts, indexes = np.unique(candidate_kpts, axis=0, return_index=True)
                fir_obs_C = fir_obs_C[indexes]
                rvec_candidate = rvec_candidate[indexes]
                tvec_candidate = tvec_candidate[indexes]
                
                # get bearing angle b/w candidates
                angles = np.zeros([candidate_kpts.shape[0],1]).astype('float32')
                for l in range(candidate_kpts.shape[0]):
                    R_first, _ = cv2.Rodrigues((rvec_candidate[l,:]).T)
                    R_now,_ = cv2.Rodrigues(rvec) 
                             
                    a1_1 = np.vstack([(fir_obs_C[l,:][None]).T,1])
                    a2_2 = np.vstack([(candidate_kpts[l,:][None]).T,1])
                    
                    # normalize
                    a1_1 = np.linalg.inv(self.K) @ a1_1
                    a2_2 = np.linalg.inv(self.K) @ a2_2
                    
                    a2 = R_now @ a2_2
                    a1 = R_first @ a1_1
                    
                    temp = ((np.dot(a1.T,a2)))/(np.linalg.norm(a1)*np.linalg.norm(a2))                    
                    
                    if temp >=1:
                        temp = 1
                    elif temp<=-1:
                        temp = -1
                        
                    angles[l] = abs(math.acos(temp))
                    
                #if that angle is above a certain threshold, add it to the good_img_keypoints2
                threshold = 10/180*np.pi
                index = np.where(angles >= threshold)
                
                added = 0
                
                for l in range(min(index[0].shape[0],300)):
                    
                    # This step is triangulation of new landmark from candidate
                    idx = index[0][l]
                    
                    R_first,_ = cv2.Rodrigues((rvec_candidate[idx,:]))
                    R_now,_ = cv2.Rodrigues(rvec)
 
                    t_first = (tvec_candidate[idx,:][None]).T                  
                                                   
                    t1 = -R_first.T @ t_first
                    t2 = -R_now.T @ tvec
                    
                    M1 = np.concatenate((R_first.T,t1), axis = 1)
                    M2 = np.concatenate((R_now.T,t2), axis = 1)   

       
                    inliers1 = np.array(fir_obs_C[idx,:])[None]
                    inliers2 = np.array(candidate_kpts[idx,:])[None]            
                  
                    inliers1 = np.vstack((inliers1.T, 1.0))
                    inliers2 = np.vstack((inliers2.T, 1.0))

                    norm_inliers1 = np.linalg.inv(self.K) @ inliers1
                    norm_inliers2 = np.linalg.inv(self.K) @ inliers2
                    
                    points3D = cv2.triangulatePoints(M1, M2, norm_inliers1[:2,:], norm_inliers2[:2,:])
                    points3D /= points3D[3]
                    
                    if (points3D[2] > 0 ):    
                        good_img_keypoints2 = np.vstack([good_img_keypoints2,candidate_kpts[idx,:]])
                        good_img_landmarks1 = np.vstack([good_img_landmarks1,(points3D[0:3]).T]) 

                        # Remove duplicate keypoints
                        good_img_keypoints2, indexes = np.unique(good_img_keypoints2, axis=0, return_index=True)
                        good_img_landmarks1 = good_img_landmarks1[indexes]
                        added = added+1
                    
                    

                
            p0 = good_img_keypoints2.reshape(-1,1,2) # P as per problem statement

            # plots candidate keypoints on left side and good keypoints in right side
            candidate_kpts_obj = self.h.kpts2kpts2Object(candidate_kpts)
            output_image1 = cv2.drawKeypoints(cv2.cvtColor(self.images[i], cv2.COLOR_GRAY2BGR), candidate_kpts_obj, 0, (0,255,255))

            good_img_kpts_obj = self.h.kpts2kpts2Object(good_img_keypoints2)
            output_image2 = cv2.drawKeypoints(cv2.cvtColor(self.images[i], cv2.COLOR_GRAY2BGR), good_img_kpts_obj, 0, (0,255,0))


            horizontal_concat = np.concatenate((output_image1, output_image2), axis=1)
            cv2.imshow('left_candidate right_initialized', horizontal_concat)
            cv2.waitKey(1)
            candidate_kpts = candidate_kpts.reshape(-1,1,2)
            
        plt.plot(T_X,T_Y)
        plt.show()

        self.h.generate_trajectory(list(zip(T_X, T_Y)))