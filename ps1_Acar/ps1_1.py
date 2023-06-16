import cv2
import numpy as np

# Load the input image
img = cv2.imread('./input/input_image.jpg')
cv2.imshow('Original Image', img)
cv2.waitKey(1000)

# Get the dimensions of the input image
height, width, channels = img.shape

# Define the center coordinates of the input image
center_x = int(width / 2)
center_y = int(height / 2)

# Define the size of the region to crop
crop_width = int(center_x / 2)
crop_height = int(center_y / 2)

# Crop out the region from the center of the input image
cropped_img = img[center_y-crop_height:center_y+crop_height, center_x-crop_width:center_x+crop_width]

# Save the cropped image as a PNG file
cv2.imwrite('./output/ps1_1_cropped_image.png', cropped_img)
cv2.imshow('Cropped Image', cropped_img)
cv2.waitKey(1000)

# Extract the red channel of the image
red_channel = cropped_img[:,:,2]
cv2.imwrite('./output/ps1_1_red_channel.png', red_channel)

# Display the red channel
cv2.imshow('Red Channel', red_channel)
cv2.waitKey(1000)

# Convert the image to grayscale
gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./output/ps1_1_gray_scaled.png', gray_img)

# Display the grayscale image
cv2.imshow('Grayscale', gray_img)
cv2.waitKey(1000)

# Define Sobel filters for x and y directions
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Apply the Sobel filters to the grayscale image
gradient_x = cv2.filter2D(gray_img, -1, sobel_x).astype(np.float32)
gradient_y = cv2.filter2D(gray_img, -1, sobel_y).astype(np.float32)

# Obtain the gradient magnitude and orientation
gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)

# Convert gradient magnitude to uint8 for display
gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
cv2.imwrite('./output/ps1_1_gradient_magnitude.png', gradient_magnitude)
gradient_orientation = cv2.phase(gradient_x, gradient_y, angleInDegrees=True)
cv2.imwrite('./output/ps1_1_gradient_orientation.png', gradient_orientation.astype('uint8'))
# Display the gradient magnitude and orientation
cv2.imshow('Gradient Magnitude', gradient_magnitude)
cv2.waitKey(1000)
cv2.imshow('Gradient Orientation', gradient_orientation.astype('uint8'))
cv2.waitKey(1000)

# Obtain Laplacian of Gaussian images for different sigma values
log_images = []
sigma_values = [1, 2, 3, 4]

for sigma in sigma_values:
    # Apply the Laplacian of Gaussian filter to the grayscale image
    log_filter = cv2.Laplacian(cv2.GaussianBlur(gray_img, (0, 0), sigma), -1, ksize=3)
    log_images.append(log_filter)

# Display the Laplacian of Gaussian images for different sigma values
for i, log_image in enumerate(log_images):
    cv2.imshow(f'LoG (Sigma={sigma_values[i]})', log_image)
    cv2.imwrite(f'./output/ps1_1_LoG_sigma_{sigma_values[i]}.png', log_image)
    cv2.waitKey(1000)
cv2.waitKey(0)