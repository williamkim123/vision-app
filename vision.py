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

diff = (diff * 255).astype("uint8")
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(first_image.shape, dtype='uint8')
filled_after = second_image.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area > 40:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(first_image, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(second_image, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(mask, [c], 0, (0,255,0), -1)
        cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

# cv2.imshow('before', first_image)
# cv2.imshow('after', second_image)
# cv2.imshow('diff',diff)
# cv2.imshow('mask',mask)
# cv2.imshow('filled after',filled_after)
# cv2.waitKey(0)

# computing the difference through subtraction

# Either coloring the difference or outputting a value that signify the difference


