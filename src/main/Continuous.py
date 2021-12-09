import numpy as np

from Image import Image
from helpers import helpers
from scipy.spatial.distance import cdist

class Continuous:
    def __init__(self, keypoints, landmarks, R, T, images):
        self.h = helpers()
        self.init_R = R
        self.init_T = T
        self.init_keypoints = keypoints
        self.init_landmarks = landmarks
        self.images = images
        self.init_keypoint_des = self.h.describe_keypoints(8, self.images[2], self.init_keypoints)
        self.images = self.images[3:]
    
    def run(self):
        for image in self.images:
            img_obj = Image(image)
            img_keypoints = img_obj.get_keypoints()
            img_des = img_obj.get_keypoints_descriptions()
            kpts_correspondence = self.get_keypoints_correspondence(self.init_keypoints, img_keypoints, self.init_keypoint_des, img_des)
            self.init_keypoints = img_keypoints
            self.init_keypoint_des = img_des
            print(len(kpts_correspondence[1]))
            
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