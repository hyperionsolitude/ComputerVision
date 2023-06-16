import cv2
import numpy as np

# Read the images
img1 = cv2.imread('input/img1.ppm')
img2 = cv2.imread('input/img2.ppm')

cv2.imwrite("input/img1.png", img1)
cv2.imwrite("input/img2.png", img2)
# Convert the images to grayscale
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#Optical Flow
flow = cv2.calcOpticalFlowFarneback(img1_gray, img2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Warp image1 onto image2 using the calculated optical flow
optical_warp = cv2.remap(img1, flow, None, cv2.INTER_LINEAR)
cv2.imwrite(f"output/optical_warp.png",optical_warp)

# Calculate the residual
optical_residual = cv2.absdiff(img2, optical_warp)
cv2.imwrite(f"output/optical_residual.png",optical_residual)

#---------------------------------------------------------------------
#Shi-tomasi

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Detect Shi-Tomasi corner points in the first image
corners1 = cv2.goodFeaturesToTrack(img1_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(img1)

# Calculate optical flow using Lucas-Kanade method
corners2, status, _ = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, corners1, None)

# Select good points
good_new = corners2[status == 1]
good_old = corners1[status == 1]

# Draw the image with keypoint tracks
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
    img1 = cv2.circle(img1, (a, b), 5, (0, 0, 255), -1)
    img2 = cv2.circle(img2, (a, b), 5, (0, 0, 255), -1)

# Combine the rotated image and the mask
result = cv2.add(img2, mask)

# Save the Shi-Tomasi corner images
cv2.imwrite("output/Shi-Tomasi1.png", img1)
cv2.imwrite("output/Shi-Tomasi2.png", img2)

# Display the image with keypoint tracks
cv2.imwrite("output/lucas.png", result)