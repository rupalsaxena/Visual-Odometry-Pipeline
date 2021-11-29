#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:40:09 2021

@author: jaswin
"""

import os
import math
import scipy.signal
import numpy as np
from PIL import Image as PImage
from matplotlib import pyplot as plt
import cmath
import random

import time

def loadImages(path):
    """
    return list of images
    """

    imagesList = []
    valid_images = (".jpg",".gif",".png",".tga")
    
    index = 0
    for file in sorted(os.listdir(path)):
        if file.endswith(valid_images) &(index <10):
            imagesList.append(file)    
            index +=1
            
    loadedImages = []
    for image in imagesList:
        img = PImage.open(path + image).convert('L')
        img = np.array(img)
        loadedImages.append(img)
        
    return loadedImages

def harris(img,patch_size,kappa):
    """
    out: matrix of Harris scores, same size as image
    """
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    Ix = scipy.signal.convolve2d(img, sobel_x, mode='valid')
    Iy = scipy.signal.convolve2d(img, sobel_y, mode='valid')
    Ixx = Ix*Ix
    Iyy = Iy*Iy
    Ixy = Ix*Iy
    
    patch = np.ones([patch_size,patch_size])
    pr = math.floor(patch_size/2)
    
    sum_Ixx = scipy.signal.convolve2d(Ixx, patch, mode='valid')
    sum_Ixy = scipy.signal.convolve2d(Ixy, patch, mode='valid')
    sum_Iyy = scipy.signal.convolve2d(Iyy, patch, mode='valid')
    
    scores = (sum_Ixx*sum_Iyy - sum_Ixy*sum_Ixy) - kappa*(sum_Ixx + sum_Iyy)*(sum_Ixx + sum_Iyy)
    scores = scores.clip(min=0)
    scores = np.pad(scores,[(1+pr,1+pr),(1+pr,1+pr)],mode='constant')
    
    return scores
    
def selectKeypoints(scores,num,r):
    """
    in: Harris scores matrix, num of selected keypoints
    r = patch radius that are zeroed around selected keypoints
    out: keypoints matrix coordinate, shape = [rows;cols], 2 by num size
    """
    keypoints = np.zeros([2,num])
    temp_scores = np.pad(scores,[(r,r),(r,r)],mode = 'constant')
    
    for i in range(num):
        max_index = np.array(np.unravel_index(temp_scores.argmax(), temp_scores.shape))
        keypoints[:,i] = max_index-r

        temp_scores[max_index[0]-r:max_index[0]+r+1,max_index[1]-r:max_index[1]+r+1] = 0
        
    return keypoints.astype(int)

def matchKeypoints(database,querry,lamb):
    """
    match querry to database,
    out: matches, array containing index of database keypoints that 
    matches the query keypoints
    ex: matches = [0,0,0,11,0,0,2,0,0,0,5,0,0,3], size = size of querry
    """
    num_q = querry.shape[1]
    matches = np.zeros(num_q)
    
    num_d = database.shape[1]
    
    d_index = np.arange(num_d)
    
    d_min = 100000
    for i in range(num_q):
        for j in range(num_d):
            dist = np.linalg.norm(database[:,j]-querry[:,i])
            if (dist < d_min) & (dist != 0):
                d_min = dist
                
    # case when all dist min = 0 (same image)            
    if d_min == 100000:
        d_min = 100
    
    for i in range(num_q):
        index = -1
        dist_prev = 100000
        
        if database.shape[1] > 1:
            for j in range(database.shape[1]):
                dist = np.linalg.norm(database[:,j]-querry[:,i])
                if (dist<dist_prev) & (dist<lamb*d_min):
                    index = j
                    dist_prev = dist
                
        if index>=0:
            matches[i] = d_index[index]
            database = np.delete(database,index,1)
            d_index = np.delete(d_index,index)
            
    return matches

def describeKeypoints(img,keypoints,r):    
    """
    vectorized patch description around keypoints, r = patch radius
    out: descriptor matrix, collumns: 'squashed'(vectorized) patch
    around each keypoints. num of rows = num of keypoints
    """
    descriptor = np.zeros([(2*r+1)*(2*r+1),keypoints.shape[1]])
    img = np.pad(img,[(r,r),(r,r)],mode = 'constant')
    
    for i in range(keypoints.shape[1]):
        row = int(keypoints[0,i])
        col = int(keypoints[1,i])
        temp_patch = img[row:row+2*r+1,col:col+2*r+1]
        temp_patch = np.reshape(temp_patch,(-1,1))
        descriptor[:,i] = temp_patch[:,0]
    return descriptor
              
def keypointsCorrespondence(keypoints1,keypoints2,matches):  
    """
    make a 2 by num_matches by 2 matrix containing matching keypoints of 2 images
    keypoints_correspondence[:,:,0] = pixel coord in img1
    keypoints_correspondence[:,:,1] = pixel coord in img2
        """
    # IMPORTANT: row and collums of keypoints are flipped here!
    # this makes pixel coordinates correct (u = col, v = row), where u is x, v is y
    
    num = np.count_nonzero(matches)
    keypoints_correspondence = np.zeros([2,num,2])
    index = 0
    
    for i in range(keypoints2.shape[1]):
        if matches[i] != 0:
            keypoints_correspondence[:,index,0] = np.flip(keypoints1[:,int(matches[i])])
            keypoints_correspondence[:,index,1] = np.flip(keypoints2[:,i])
            
            index = index+1   
            
    return keypoints_correspondence

def reshuffleLandmark(landmark,matches):
    
    """
    Only used in localization()
    reshuffle the landmark so that it corresponds to match b/w keypoints_prev and keypoints_new
    resulting from keypoints_correspondence()
    out: 4 by N matrix (X,Y,Z,1).T
    """
    num = np.count_nonzero(matches)
    reshuffled_landmark = np.zeros([landmark.shape[0],num])
    index = 0
    
    for i in range(matches.shape[0]):
        if matches[i] !=0:
            reshuffled_landmark[:,index] = landmark[:,int(matches[i])]
            index = index + 1

    return(reshuffled_landmark)

def eightPoints(keypoints_correspondence, K): 
    """
    keypoints_correspondence[:,:,0] = pixel coord in img1
    keypoints_correspondence[:,:,1] = pixel coord in img2
    out:Essential matrix
    """
    
    num = keypoints_correspondence.shape[1]
    
    p1_set = np.ones([3,num])
    p2_set = np.ones([3,num])
    
    p1_set[0:2,:] = keypoints_correspondence[:,:,0]
    p2_set[0:2,:] = keypoints_correspondence[:,:,1]
    
    # normalize p1_set and p2_set
    p1_mean = (np.mean(p1_set,axis = 1))[:,None]
    p2_mean = (np.mean(p2_set,axis = 1))[:,None]
    
    p1_sigma = np.linalg.norm(p1_set - p1_mean)/num
    p2_sigma = np.linalg.norm(p2_set - p2_mean)/num
    
    s1 = 1/p1_sigma*math.sqrt(2)
    s2 = 1/p2_sigma*math.sqrt(2)
    
    T1 = np.array([[s1,0,-s1*p1_mean[0,0]],[0, s1, -s1*p1_mean[1,0]],[0,0,1]])
    T2 = np.array([[s2,0,-s2*p2_mean[0,0]],[0, s2, -s2*p2_mean[1,0]],[0,0,1]])
    
    p1_set = T1 @ p1_set
    p2_set = T2 @ p2_set
    
    # make Q matrix
    Q = np.zeros([num,9])
    for i in range(num):
        p1 = p1_set[:,i]
        p2 = p2_set[:,i]
        Q[i,:] = np.kron(p1,p2)
        
    u, s, vh = np.linalg.svd(Q)
    v = vh.T
    
    F = v[:,v.shape[1]-1]
    F = F[:,None]
    
    F = np.reshape(F,(3,3))
    F = F.T
    
    # unormalize F
    F = T2.T @ F @ T1
    
    # enforce det(F) = 0
    u,s,vh = np.linalg.svd(F)
    s[2] = 0
    S = np.diag(s)
    F = u @ S @ vh
    
    # extract essential matrix
    E = K.T @ F @ K
    return E
    
def triangulate(M1,M2,keypoints_correspondence):
        
    """
    keypoints_correspondence[:,:,0] = pixel coord in img1
    keypoints_correspondence[:,:,1] = pixel coord in img2
    return landmark points P, 4 by N matrix (X,Y,Z,1)
        
    """
    num = keypoints_correspondence.shape[1]
    
    p1_set = np.ones([3,num])
    p2_set = np.ones([3,num])
    
    p1_set[0:2,:] = keypoints_correspondence[:,:,0]
    p2_set[0:2,:] = keypoints_correspondence[:,:,1]    
    
    P = np.zeros([4,num])
    
    for i in range(num):
        p1 = p1_set[:,i]
        p2 = p2_set[:,i]
        
        p1_skew = np.array([[0,-p1[2],p1[1]],[p1[2],0,-p1[0]],[-p1[1],p1[0],0]])
        p2_skew = np.array([[0,-p2[2],p2[1]],[p2[2],0,-p2[0]],[-p2[1],p2[0],0]])
        
        A_top = p1_skew @ M1
        A_bottom = p2_skew @ M2
        
        A = np.concatenate((A_top,A_bottom),axis = 0)
        
        u,s,vh = np.linalg.svd(A)
        
        v = vh.T
        
        P_temp = v[:,v.shape[1]-1]
        P_temp = P_temp/(P_temp[3])
        
        P[:,i] = P_temp
    
    return P

def decomposeEssential(E,K,keypoints_correspondence):
    
    """
    Decompose essential matrix E and CHOOSES one correct R and T
    """
    u, s, vh = np.linalg.svd(E)
    
    w = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    u3 = u[:,2]
    u3 = u3[:,None]
    
    r1 = u @ w @ vh
    if np.linalg.det(r1)<=0:
        r1 = -r1
        
    r2 = u @ w.T @ vh    
    if np.linalg.det(r2)<=0:
        r2 = -r2
    
    #Uses axis-angle representation of rotation
    #choose rotation with smaller angles
    
    val2 = (r2[0,0] + r2[1,1] + r2[2,2] - 1)/2
    if abs(val2) > 1:
        val2 = math.copysign(1,val2)       
    theta2 = 180/math.pi*math.acos(val2)
    
    val1 = (r1[0,0] + r1[1,1] + r1[2,2] - 1)/2
    if abs(val1) > 1:
        val1 = math.copysign(1,val1)
    theta1 = 180/math.pi*math.acos(val1)
    
    # print(u3)
    
    if abs(theta1) > abs(theta2):
        r = r2
    else:
        r = r1
    
    # print(theta1,theta2)
    # print(r1,r2)
    
    m2_1 = K @ np.concatenate((r,u3),axis = 1)
    m2_2 = K @ np.concatenate((r,-u3),axis = 1)
    
    m1 = K @ np.concatenate((np.eye(3),np.zeros(3)[:,None]),axis = 1)
  
    P1 = triangulate(m1, m2_1, keypoints_correspondence)
    P2 = triangulate(m1, m2_2, keypoints_correspondence)
  
    num1 = 0
    num2 = 0
    # number of tringulated points with positive Z
    for i in range(P1.shape[1]):

        if (P1[2,i] > 0):
            num1 += 1
         
        if (P2[2,i] > 0):
            num2 += 1  

    max_num = max(num1,num2) 
    
    if (max_num == num1):
        R = r
        T = u3
    if (max_num == num2):
        R = r
        T = -u3
        
    # print(num1,num2)
        
    R[:,0] = R[:,0]/np.linalg.norm(R[:,0])
    R[:,1] = R[:,1]/np.linalg.norm(R[:,1])
    R[:,2] = R[:,2]/np.linalg.norm(R[:,2])
        
    return R,T

def decomposeEssentialAmbiguous(E):
   
    """
    Decompose essential matrix E and DONT CHOOSE correct T
    T here can be -u3 or u3. needs to be chosen with ransac
    """  
    u, s, vh = np.linalg.svd(E)
    
    w = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    u3 = u[:,2]
    u3 = u3[:,None]
    
    r1 = u @ w @ vh
    if np.linalg.det(r1)<=0:
        r1 = -r1
        
    r2 = u @ w.T @ vh    
    if np.linalg.det(r2)<=0:
        r2 = -r2
        
    val2 = (r2[0,0] + r2[1,1] + r2[2,2] - 1)/2
    if abs(val2) > 1:
        val2 = math.copysign(1,val2)       
    theta2 = 180/math.pi*math.acos(val2)
    
    val1 = (r1[0,0] + r1[1,1] + r1[2,2] - 1)/2
    if abs(val1) > 1:
        val1 = math.copysign(1,val1)
    theta1 = 180/math.pi*math.acos(val1)
    
    # print(u3)
    
    if abs(theta1) > abs(theta2):
        R = r2
    else:
        R = r1
    
    # print(theta1,theta2)
    # print(r1,r2)        
        
    R[:,0] = R[:,0]/np.linalg.norm(R[:,0])
    R[:,1] = R[:,1]/np.linalg.norm(R[:,1])
    R[:,2] = R[:,2]/np.linalg.norm(R[:,2])
    T = u3
        
    return R,T

def reproject(P,M,p):
   
    """
    find reprojection error
    P = [X,Y,Z,1].T
    p = [u,v]
    M = K[R|T]
    """
   
    p = p.flatten()
    p_rep = M @ P
    
    p_rep = p_rep.flatten()
    p_rep = p_rep/p_rep[2]
    
    err = np.linalg.norm(p_rep - p)
    # print(p,p_rep,err)
    return err
    
def p3p(worldPoints, imageVectors):
    
    """
    find 4 poses as solution of p3p
    if no solution: return all zeros in poses
    poses[:,:,0] = [R1|T1], and so on
    
    """
    poses = np.zeros([3,4,4])
    # print("worldpoints")
    # print(worldPoints)
    # print("imagevec")
    # print(imageVectors)
    
    P1 = worldPoints[:,0]
    P2 = worldPoints[:,1]
    P3 = worldPoints[:,2]
    
    P1 = P1[:,None]
    P2 = P2[:,None]
    P3 = P3[:,None]
    
    vector1 = P2 - P1
    vector2 = P3 - P1
    

    if (np.linalg.norm(np.cross(vector1.T,vector2.T)) == 0):
        print("vectors collinear")
        return poses
    
    f1 = imageVectors[:,0][:,None]
    f2 = imageVectors[:,1][:,None]
    f3 = imageVectors[:,2][:,None]

    e1 = f1.T
    e3 = np.cross(f1.T,f2.T)
    e3 = e3/np.linalg.norm(e3)

    e2 = np.cross(e3,e1)  
    
    T = np.concatenate((e1[None,:],e2[None,:],e3[None,:]),axis = 0)
    T = T[:,0,:]


    f3 = T @ f3
        
    if (f3[2,0] > 0):     
        f1 = imageVectors[:,1][:,None]
        f2 = imageVectors[:,0][:,None]
        f3 = imageVectors[:,2][:,None]

        e1 = f1.T
        e3 = np.cross(f1.T,f2.T)
        e3 = e3/np.linalg.norm(e3)

        e2 = np.cross(e3,e1)
    
        T = np.concatenate((e1[None,:],e2[None,:],e3[None,:]),axis = 0)
        T = T[:,0,:]

        f3 = T @ f3
        
        P1 = worldPoints[:,1]
        P2 = worldPoints[:,0]
        P3 = worldPoints[:,2]
        
        P1 = P1[:,None]
        P2 = P2[:,None]
        P3 = P3[:,None]
        
    n1 = (P2-P1).T
    n1 = n1/np.linalg.norm(n1)
    n3 = np.cross(n1,(P3-P1).T)
    n3 = n3/np.linalg.norm(n3)
    n2 = np.cross(n3,n1)
    
    N = np.concatenate((n1[None,:],n2[None,:],n3[None,:]),axis = 0)
    N = N[:,0,:]
    
    P3 = N @ (P3-P1)
    
    d_12 = np.linalg.norm(P2 - P1)
    f_1 = f3[0,0]/f3[2,0]
    f_2 = f3[1,0]/f3[2,0]
    p_1 = P3[0,0]
    p_2 = P3[1,0]
    
    cos_beta = float(f2.T @ f1)
    b = 1/(1-cos_beta*cos_beta) - 1
    
    if cos_beta < 0:
        b = -math.sqrt(b)
    else:
        b = math.sqrt(b)    
    
    f_1_pw2 = f_1*f_1
    f_2_pw2 = f_2*f_2
    p_1_pw2 = p_1*p_1
    p_1_pw3 = p_1_pw2 * p_1
    p_1_pw4 = p_1_pw3 * p_1
    p_2_pw2 = p_2*p_2
    p_2_pw3 = p_2_pw2 * p_2
    p_2_pw4 = p_2_pw3 * p_2
    d_12_pw2 = d_12*d_12
    b_pw2 = b*b
    
    factor_4 = (-f_2_pw2*p_2_pw4
               -p_2_pw4*f_1_pw2
               -p_2_pw4)

    factor_3 = (2*p_2_pw3*d_12*b
               +2*f_2_pw2*p_2_pw3*d_12*b
               -2*f_2*p_2_pw3*f_1*d_12)

    factor_2 = (-f_2_pw2*p_2_pw2*p_1_pw2 
               -f_2_pw2*p_2_pw2*d_12_pw2*b_pw2 
               -f_2_pw2*p_2_pw2*d_12_pw2 
               +f_2_pw2*p_2_pw4 
               +p_2_pw4*f_1_pw2 
               +2*p_1*p_2_pw2*d_12 
               +2*f_1*f_2*p_1*p_2_pw2*d_12*b 
               -p_2_pw2*p_1_pw2*f_1_pw2 
               +2*p_1*p_2_pw2*f_2_pw2*d_12 
               -p_2_pw2*d_12_pw2*b_pw2 
               -2*p_1_pw2*p_2_pw2)

    factor_1 = (2*p_1_pw2*p_2*d_12*b 
               +2*f_2*p_2_pw3*f_1*d_12 
               -2*f_2_pw2*p_2_pw3*d_12*b 
               -2*p_1*p_2*d_12_pw2*b)

    factor_0 = (-2*f_2*p_2_pw2*f_1*p_1*d_12*b 
               +f_2_pw2*p_2_pw2*d_12_pw2 
               +2*p_1_pw3*d_12 
               -p_1_pw2*d_12_pw2 
               +f_2_pw2*p_2_pw2*p_1_pw2 
               -p_1_pw4 
               -2*f_2_pw2*p_2_pw2*p_1*d_12 
               +p_2_pw2*f_1_pw2*p_1_pw2 
               +f_2_pw2*p_2_pw2*d_12_pw2*b_pw2)
    
    # print("factor")
    # print(factor_4,factor_3,factor_2,factor_1,factor_0)
    
    x = solveQuartic(factor_4,factor_3,factor_2,factor_1,factor_0)
    
    # print("x")
    # print(x)
    
    for i in range(4):
        
        
        if abs(x[i].real) < 1:
            cot_alpha = (-f_1*p_1/f_2-(x[i].real)*p_2+d_12*b)/(-f_1*(x[i].real)*p_2/f_2+p_1-d_12)
            cos_theta = x[i].real
            sin_theta = math.sqrt(1-x[i].real*x[i].real)
            sin_alpha = math.sqrt(1/(cot_alpha*cot_alpha+1))
            cos_alpha = math.sqrt(1-sin_alpha*sin_alpha)
        
            if cot_alpha < 0:
                cos_alpha = -cos_alpha
            
            C = np.array([[d_12*cos_alpha*(sin_alpha*b+cos_alpha)],
              [cos_theta*d_12*sin_alpha*(sin_alpha*b+cos_alpha)],
              [sin_theta*d_12*sin_alpha*(sin_alpha*b+cos_alpha)]])
        
            C = P1 + N.T @ C
        
            R = np.array([[-cos_alpha,-sin_alpha*cos_theta,-sin_alpha*sin_theta],
              [sin_alpha,-cos_alpha*cos_theta,-cos_alpha*sin_theta],
              [0,-sin_theta,cos_theta]])
        
            R = N.T @ R.T @ T
        else:
            R = np.zeros([3,3])
            C = np.zeros([3,1])
            
        C = -R.T @ C
        R = R.T
        poses[:,0:3,i] = R
        poses[:,3,i] = C.T
        
    # print("poses")
    # print(poses[:,:,0])
    # print(poses[:,:,1])   
    # print(poses[:,:,2])
    # print(poses[:,:,3])
    return poses
        
def solveQuartic(A,B,C,D,E):        
    """
    Used only in p3p
    """
    A_pw2 = A*A
    B_pw2 = B*B
    A_pw3 = A_pw2*A
    B_pw3 = B_pw2*B
    A_pw4 = A_pw3*A
    B_pw4 = B_pw3*B
    
    alpha = -3*B_pw2/(8*A_pw2)+C/A
    beta = B_pw3/(8*A_pw3)-B*C/(2*A_pw2)+D/A
    gamma = -3*B_pw4/(256*A_pw4)+B_pw2*C/(16*A_pw3)-B*D/(4*A_pw2)+E/A
    
    alpha_pw2 = alpha*alpha
    alpha_pw3 = alpha_pw2*alpha
    
    P = -alpha_pw2/12-gamma
    Q = -alpha_pw3/108+alpha*gamma/3-beta*beta/8
    R = -Q/2+cmath.sqrt(Q*Q/4+P*P*P/27)
    U = R**(1/3)
    
    if U == 0:
        y = -5*alpha/6-Q**(1/3)
    else:
        y = -5*alpha/6-P/(3*U)+U
    
    w = cmath.sqrt(alpha+2*y)
    
    roots = np.zeros(4,dtype=np.complex_)
    roots[0] = -B/(4*A) + 0.5*(w+cmath.sqrt(-(3*alpha+2*y+2*beta/w)))
    roots[1] = -B/(4*A) + 0.5*(w-cmath.sqrt(-(3*alpha+2*y+2*beta/w)))
    roots[2] = -B/(4*A) + 0.5*(-w+cmath.sqrt(-(3*alpha+2*y-2*beta/w)))
    roots[3] = -B/(4*A) + 0.5*(-w-cmath.sqrt(-(3*alpha+2*y-2*beta/w)))  
      
    return roots       
 

def initialization(image1,image2,K,num_iter):   
    
    """
    Return: inlier_keypoints_best = keypoints correspondece that is in inlier 
    shape: 2 by N by 2 (points in image1 = [:,:,0])
    
    inlier_landmark_best = 4 by N 3d coord
    
    best pose:
        
    R_best  T_best
    """
    
    harris_scores1 = harris(image1,9,0.08)
    import pdb; pdb.set_trace()
    harris_keypoints1 = selectKeypoints(harris_scores1,300,8) 
    keypoints_des1 = describeKeypoints(image1, harris_keypoints1, 8)
 
    harris_scores2 = harris(image2,9,0.08)
    harris_keypoints2 = selectKeypoints(harris_scores2,300,8) 
    keypoints_des2 = describeKeypoints(image2, harris_keypoints2, 8)   
   
    keypoints_match = matchKeypoints(keypoints_des1, keypoints_des2, 4)
    keypoints_correspondence = keypointsCorrespondence(harris_keypoints1, harris_keypoints2, keypoints_match)
     
    # print(keypoints_correspondence.shape)
    # print(keypoints_correspondence[:,:,0])
    # print(keypoints_correspondence[:,:,1])
    
    # keypoints_correspondence = testin()
    # K = np.array([[1379.74,0,760.35],
    # [0,1382.08,503.41],
    # [0,0,1]])
    num = keypoints_correspondence.shape[1] 
    num_best = -1
    
    inlier_keypoints_best = []
    inlier_landmark_best = []
    
    R_best = np.zeros([3,3])
    T_best = np.zeros([3,1])
    
    # RANSAC 
    for i in range(num_iter):
        
        inlier_keypoints = []
        inlier_landmark = []
        inlier_keypoints1 = []
        inlier_landmark1 = []
        inlier_keypoints2 = []
        inlier_landmark2 = []
        
        chosen_num = random.sample(range(0,num),8)
        chosen_keypoints = keypoints_correspondence[:,chosen_num,:]
    
        essential_mat = eightPoints(chosen_keypoints, K)   
   
        R,T = decomposeEssentialAmbiguous(essential_mat)
   
        M2_1 = K @ np.concatenate((R,T),axis=1)
        M2_2 = K @ np.concatenate((R,-T),axis=1)
        M1 = K @ np.concatenate((np.eye(3),np.zeros(3)[:,None]),axis = 1)
        triangulated1 = triangulate(M1, M2_1, keypoints_correspondence)
        triangulated2 = triangulate(M1, M2_2, keypoints_correspondence)    
        
        num_inlier = 0
        num_inlier1 = 0
        num_inlier2 = 0
        
        for i in range(triangulated1.shape[1]):       
            P = triangulated1[:,i]
            p = keypoints_correspondence[:,i,1]
            p = np.concatenate((p,np.array([1.0])))
            err = reproject(P, M2_1, p)
            if (err<=10)&(P[2]>0):
                num_inlier1+=1
                inlier_keypoints1.append(keypoints_correspondence[:,i,:])
                inlier_landmark1.append(P)
                
        for i in range(triangulated2.shape[1]):       
            P = triangulated2[:,i]
            p = keypoints_correspondence[:,i,1]
            p = np.concatenate((p,np.array([1.0])))
            err = reproject(P, M2_2, p)
            if (err<=10)&(P[2]>0):
                num_inlier2+=1
                inlier_keypoints2.append(keypoints_correspondence[:,i,:])
                inlier_landmark2.append(P)
                
        if num_inlier1 > num_inlier2:
            inlier_keypoints = inlier_keypoints1
            inlier_landmark = inlier_landmark1
            num_inlier = num_inlier1
            T = T
            
        else:
            inlier_keypoints = inlier_keypoints2
            inlier_landmark = inlier_landmark2
            num_inlier = num_inlier2
            T = -T
                
        
        if num_inlier > num_best:
            num_best = num_inlier
            R_best = R
            T_best = T
            inlier_keypoints_best = inlier_keypoints
            inlier_landmark_best = inlier_landmark
            
    inlier_keypoints_best = np.array(inlier_keypoints_best)
    inlier_landmark_best = np.array(inlier_landmark_best)
    
    inlier_keypoints_best = np.transpose(inlier_keypoints_best,(1,0,2))
    inlier_landmark_best = np.transpose(inlier_landmark_best,(1,0))
    
    essential_mat = eightPoints(inlier_keypoints_best, K)   
   
    R_best,T_best = decomposeEssential(essential_mat, K, inlier_keypoints_best)
    
    # print(inlier_keypoints_best.shape)
    # print(inlier_landmark_best)
    # print(R_best)
    # print(T_best)
    
    return(inlier_keypoints_best,inlier_landmark_best,R_best,T_best)
            
def ransacLocalization(landmark,keypoints,K,num_iter):
    
    """
    Use p3p to pick R,T that best fit:
    a given set of landmark coordinates (landmark)
    and corresponding keypoints (keypoints)
    """
     
    num = keypoints.shape[1]
    inv_K = np.linalg.inv(K)
 
    num_best = -1
    
    for i in range(num_iter):
        inlier_keypoints = []
        inlier_landmark = []
    
        inlier_keypoints1 = []
        inlier_landmark1 = []
    
        inlier_keypoints2 = []
        inlier_landmark2 = []
    
        inlier_keypoints3 = []
        inlier_landmark3 = []
    
        inlier_keypoints4 = []
        inlier_landmark4 = []
         
        num_chosen = random.sample(range(0,num),3)
        chosen_keypoints = keypoints[:,num_chosen]        
        chosen_keypoints = np.concatenate((chosen_keypoints,np.array([[1.0,1.0,1.0]])),axis = 0)
        
        chosen_landmark = landmark[:,num_chosen]
                
        worldPoints = chosen_landmark[0:3,:]
        
        imageVectors = inv_K @ chosen_keypoints
        imageVectors[:,0] = imageVectors[:,0]/np.linalg.norm(imageVectors[:,0])
        imageVectors[:,1] = imageVectors[:,1]/np.linalg.norm(imageVectors[:,1])
        imageVectors[:,2] = imageVectors[:,2]/np.linalg.norm(imageVectors[:,2])                        
        
        poses = p3p(worldPoints, imageVectors)
        
        r1 = poses[:,0:3,0]
        t1 = poses[:,3,0][:,None]
        
        r2 = poses[:,0:3,1]
        t2 = poses[:,3,1][:,None]
        
        r3 = poses[:,0:3,2]
        t3 = poses[:,3,2][:,None]
        
        r4 = poses[:,0:3,3]
        t4 = poses[:,3,3][:,None]
        
        M2_1 = K @ np.concatenate((r1,t1),axis=1)
        M2_2 = K @ np.concatenate((r2,t2),axis=1)
        M2_3 = K @ np.concatenate((r3,t3),axis=1)
        M2_4 = K @ np.concatenate((r4,t4),axis=1)         
        
        num_inlier = 0
        num_inlier1 = 0
        num_inlier2 = 0
        num_inlier3 = 0
        num_inlier4 = 0
        
        for i in range(num):
            P = landmark[:,i]
            p = keypoints[:,i]
            p = np.concatenate((p,np.array([1.0])))
            err1 = reproject(P, M2_1, p)
            if (err1<=10)&(P[2]>0):
                num_inlier1+=1
                inlier_keypoints1.append(keypoints[:,i])
                inlier_landmark1.append(P)
                
        for i in range(num):
            P = landmark[:,i]
            p = keypoints[:,i]
            p = np.concatenate((p,np.array([1.0])))
            err2 = reproject(P, M2_2, p)
            if (err2<=10)&(P[2]>0):
                num_inlier2+=1
                inlier_keypoints2.append(keypoints[:,i])
                inlier_landmark2.append(P)
                
        for i in range(num):
            P = landmark[:,i]
            p = keypoints[:,i]
            p = np.concatenate((p,np.array([1.0])))
            err3 = reproject(P, M2_3, p)
            if (err3<=10)&(P[2]>0):
                num_inlier3+=1
                inlier_keypoints3.append(keypoints[:,i])
                inlier_landmark3.append(P)
                
        for i in range(num):
            P = landmark[:,i]
            p = keypoints[:,i]
            p = np.concatenate((p,np.array([1.0])))
            err4 = reproject(P, M2_4, p)
            if (err4<=10)&(P[2]>0):
                num_inlier4+=1
                inlier_keypoints4.append(keypoints[:,i])
                inlier_landmark4.append(P)
        # print(num_inlier1,num_inlier2,num_inlier3,num_inlier4)
        num_inlier = max(num_inlier1,num_inlier2,num_inlier3,num_inlier4)
        
        if num_inlier == num_inlier1:
            inlier_keypoints = inlier_keypoints1
            inlier_landmark = inlier_landmark1
            R = r1
            T = t1
        
        if num_inlier == num_inlier2:
            inlier_keypoints = inlier_keypoints2
            inlier_landmark = inlier_landmark2
            R = r2
            T = t1
            
        if num_inlier == num_inlier3:
            inlier_keypoints = inlier_keypoints3
            inlier_landmark = inlier_landmark3
            R = r3
            T = t3
            
        if num_inlier == num_inlier4:
            inlier_keypoints = inlier_keypoints4
            inlier_landmark = inlier_landmark4
            R = r4
            T = t4
            
        if num_inlier > num_best:
            num_best = num_inlier
            inlier_keypoints_best = inlier_keypoints
            inlier_landmark_best = inlier_landmark
            R_best = R
            T_best = T
          
    inlier_keypoints_best = (np.array(inlier_keypoints_best)).T
    inlier_landmark_best = (np.array(inlier_landmark_best)).T
                      
    return (inlier_keypoints_best,inlier_landmark_best,R_best,T_best)
        

def localization(image1,image2,keypoints_prev,landmark_prev,K):
    
    """
    given 2 images,
    keypoints_prev = keypoints in image1
    landmark_prev = landmarks corresponding to those keypoints
    
    step: find matches b/w keypoints_prev and keypoints in image2
    using feature matching
    
    then, match keypoints in image2 to landmark_prev
    
    do ransaclocalization (ransac with p3p)
    
    done
    """
    harris_scores1 = harris(image1,9,0.08)
    # remember to flip row and col in keypoints_prev
    harris_keypoints1 = np.flip(keypoints_prev,axis = 0) 
    keypoints_des1 = describeKeypoints(image1, harris_keypoints1, 8)

   
    harris_scores2 = harris(image2,9,0.08)
    harris_keypoints2 = selectKeypoints(harris_scores2,300,8) 
    keypoints_des2 = describeKeypoints(image2, harris_keypoints2, 8)   
   
    keypoints_match = matchKeypoints(keypoints_des1, keypoints_des2, 5)
    # print(keypoints_match)
    keypoints_correspondence = keypointsCorrespondence(harris_keypoints1, harris_keypoints2, keypoints_match)

    landmark = reshuffleLandmark(landmark_prev,keypoints_match)
    
    keypoints_new = keypoints_correspondence[:,:,1]
    
    inlier_keypoints,inlier_landmark,R,T = ransacLocalization(landmark, keypoints_new, K, 300)
    
    # print(inlier_keypoints.shape)
    # print(inlier_keypoints)
    # print(inlier_landmark)
    # print(R)
    # print(T)    
    return(inlier_keypoints,inlier_landmark,R,T)

def testin():
    """
    this is values of exercise 6 matlab
    see if it gives same results
    usage = keypoints_correspondence = testin()
    K = copy from matlab file
    """
    a = np.array([694.95,699,567,1141,1337.3,370,620,1081,1215,796.95,1215,907.01,1020,1110,1000.2,150,754.94,1281,1000.1,170,1056,1304,776,1024.5,513,212.04,401.99,351,1291,909,439,277.01,783,1333.9,753.08,610.78,863.32,1264,1276,1164,1293,683,262,307.91,368.87,373,384,383.76,727.98,1173.4,1248,1037,1358.9,1308,762,683,1201,425,557.98,705.55,816.05,889.95,1382,425.62,1272,1009,811.02,204,1039,713.75,1236,664.5,1045,1384.3,853.92,584.75,1414,1033.2,537,1219.1,199.94,704.61,571.5,1382.5])    
    b = np.array([44,46,54,74,77.281,79.995,84,89.974,97,102.05,109.02,114,134,152,155.23,163.02,173.25,176,186,205,215,226,254,273.43,275.03,284,286.01,316.99,329,331,346,359,373,374,390.03,392.76,422.22,434.99,436.02,439.98,439,440.01,453.07,479,492.13,499,504,508,518,520.64,526,536,570.1,579,601,605,609,619.4,621,622.13,623,624,624,641.44,646,649,650.02,723,749,757,771.99,784,814,828.35,837.31,839,839.75,857,889,266.91,598.27,643,763.5,851.33])

    c = np.array([732.97,737,601.99,1149,1327.1,346.01,620,1092,1217.7,836,1218,925,1035,1120,1016.1,84,796.19,1278,1016,109,1070,1298,818.99,1040,523.01,162,386,327,1287,939,444,240,819,1326,788.15,637.07,891.78,1264,1275,1173,1289.9,716.35,221.99,277,362,367,378,378,767,1181.6,1250,1070,1349,1303,806,727,1208,443.88,593,750.58,865.01,939.96,1370,444,1272,1055.8,861,155.48,1087,854.75,1240,803,1094.3,1376.6,980.54,716,1404,1087,671,1222.2,148.73,749.78,703.5,1376.2])
    d = np.array([27.973,31,28.99,89,104,41.996,65,100.98,115,94,127.02,114,140,163,161,118,164.9,194,190,165.73,221,242,248.01,278,260,256.01,266.34,298,340,331,332,341,370,383,386.65,386.7,421.48,439.99,441.12,442.99,444.09,436,444.02,473,488.06,496,501,504.73,517,521.09,526,536,567.98,576.13,602,607,606,625.28,625,623.88,623.94,624,618,649,640,647,651.07,743.48,745,764,761,794,808,808.7,839.37,854,818.75,851,908,277.26,604.34,646.08,775.5,831.06])
    
    num = a.shape[0]
    
    kr = np.zeros([2,num,2])
    kr[0,:,0] = a
    kr[1,:,0] = b
    kr[0,:,1] = c
    kr[1,:,1] = d
    
    return kr
   
def main():
   start_time = time.time()

   # Load training data
   data_dir = os.getcwd()
   output_dir = os.getcwd()
   img_dir = os.path.join(data_dir, 'data/parking/images/')
   K_file = os.path.join(data_dir, 'data/parking/K.txt')
   # your images in a List
   imgs = loadImages(img_dir)
 
   with open(K_file) as f:
     lines = f.readlines()     
     lines = [line.strip(", \n") for line in lines]
     K = np.genfromtxt(lines, dtype = float, delimiter = ", ")         
     
   inlier_keypoints,inlier_landmark,R,T = initialization(imgs[0], imgs[5], K, 1200)      
   print("initialization done")
   print("--- %s seconds ---" % (time.time() - start_time))
   print("rot and trans:")
   print(R)
   print(T)
     
   # if u wanna check correctness, try img[5] , imgs[0] back to origin
   inlier_keypoints,inlier_landmark,R,T = localization(imgs[5], imgs[6], inlier_keypoints[:,:,1], inlier_landmark, K)   
   print("first localization")
   print("--- %s seconds ---" % (time.time() - start_time))
   print("rot and trans:")
   print(R)
   print(T)

   print("remember that this is R_CW and T_CW. Actual car movement is inverse of this (R = R_CW.T, T = R_CW.T @ T_CW)")  
       
if __name__ == "__main__":
    main()