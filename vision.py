import numpy as np
import cv2
from skimage.metrics import structural_similarity
from PIL import Image

# loading images in any format
# - There could be multiple images that the user want to check
first_image = cv2.imread("c:\\Users\\kyung\\Downloads\\image1.png")
second_image = cv2.imread("c:\\Users\\kyung\\Downloads\\image2.png")

def resize_images(images):
    up_width = 600
    up_height = 400
    up_points = (up_width, up_height)
    # resize the image
    resized_up = cv2.resize(images, up_points, interpolation = cv2.INTER_LINEAR)
    
    return resized_up

first_image = resize_images(first_image)
second_image = resize_images(second_image)

first_gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
second_gray = cv2.cvtColor(second_image, cv2.COLOR_BGR2GRAY)

(score, diff) = structural_similarity(first_gray, second_gray, full=True)
print("Image similarity", score)

# computing the difference through subtraction

# Either coloring the difference or outputting a value that signify the difference


