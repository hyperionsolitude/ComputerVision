import cv2
import numpy as np
import os
# Step (a): Estimate the distortion parameters and undistort the images

# Specify the directory where the images are located
image_dir = './input/'

# Get a list of all files in the image directory
image_files = os.listdir(image_dir)

# Filter the files to only include TIFF images
image_paths = [os.path.join(image_dir, file) for file in image_files if file.lower().endswith('.tif')]

# Set the size of the checkerboard squares in mm
square_size = 30

# Prepare object points (coordinates of the checkerboard corners in the real world)
pattern_size = (8, 6)  # Number of inner corners in the checkerboard pattern
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size
obj_points = []  # 3D points of the checkerboard corners in the real world
img_points = []  # 2D points of the checkerboard corners in the images

img = cv2.imread(image_paths[0])
cv2.imshow('IMG', img)
cv2.waitKey(0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)
cv2.waitKey(0)