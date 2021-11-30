from Image import Image
class Initialization:
    def __init__(self, img1, img2, K, num_iter=1200):
        self.image1 = img1
        self.image2 = img2
        self.K = K
        self.num_iter = num_iter
        
    def generate_keypoints_correspondences(self):
        img_obj1 = Image(self.image1)
        img_obj2 = Image(self.image2)

        keypoints1 = img_obj1.get_keypoints()
        keypoints2 = img_obj2.get_keypoints()

        keypoint_des1 = img_obj1.get_keypoints_descriptions()
        keypoint_des2 = img_obj2.get_keypoints_descriptions()

        keypoints_match = self.match_keypoints(keypoint_des1, keypoint_des2)
    
    def match_keypoints(self, keypoint_des1, keypoint_des2):
        # Thought! Maybe match keypoints should be a class or a function in helpers function 
        # since it can be used during localization as well
        # TODO: implement this method
        return None