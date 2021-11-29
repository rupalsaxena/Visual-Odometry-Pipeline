from Image import Image
class Initialization:
    def __init__(self, img1, img2, K, num_iter=1200):
        self.img1 = Image(img1)
        self.img2 = Image(img2)
        self.K = K
        self.num_iter = num_iter
        