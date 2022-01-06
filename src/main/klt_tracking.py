import numpy as np
import cv2
import argparse
from helpers import helpers
import numpy as np

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.05,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (49,49),
                  maxLevel = 7,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = [255,0,0]
# Take first frame and find corners in it
IMAGE_PATH = 'data/malaga-urban-dataset-extract-07/left_800x600/'
images = helpers().loadImages(IMAGE_PATH)

images = list(map(np.uint8, images))
old_frame = images[0]

old_gray = old_frame
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
print(p0.shape)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
# import pdb; pdb.set_trace() 
for i in range(1, len(images)):

    frame = images[i]
    frame_gray = frame
    # if int(i)%50 == 0: # retrack new points
    #     p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    #     mask = np.zeros_like(old_frame)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color, 1)
        frame = cv2.circle(frame,(int(a),int(b)),5,color,-1)
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)
    cv2.waitKey(100)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    # print(p0.shape)
    # p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
    # print(p0.shape)
    # break