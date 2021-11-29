import cv2 

class Image:
    def __init__(self, image):
        self.image = image

        self.harris_score = self.generate_harris_score()
        self.harris_keypoints = None
        self.keypoints_description = None

        
    
    def generate_harris_score(self):
        # TODO: Tune parameters
        harris_score = cv2.cornerHarris(self.image,2,3,0.04)
        return harris_score

