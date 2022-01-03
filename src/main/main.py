import os
import time
from pipeline import Pipeline


# IMAGE_PATH = "data/parking/images/"
# K_PATH = "data/parking/K.txt"

IMAGE_PATH = 'data/malaga-urban-dataset-extract-07/left_800x600/'
K_PATH = 'data/malaga-urban-dataset-extract-07/k_800x600_left.txt'

def main():
    start_time = time.time()

    current_path = os.getcwd()

    img_dir = os.path.join(current_path, IMAGE_PATH)
    K_file = os.path.join(current_path, K_PATH)

    pipeline = Pipeline(img_dir, K_file)
    pipeline.run()

    print(f"runtime in seconds: {time.time() - start_time}")



if __name__ == "__main__":
    main()