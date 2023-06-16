import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt

class HoughTransform:
    def __init__(self, theta_resolution=1, rho_resolution=1):
        self.theta_resolution = theta_resolution
        self.rho_resolution = rho_resolution

    def hough_lines_acc(self, edge_image):
        # Get the height and width of the edge image
        height, width = edge_image.shape

        # Calculate the maximum distance between a point in the image and the origin of the Hough space
        diagonal = int(np.sqrt(height**2 + width**2))

        # Generate an array of possible values for the rho parameter in the Hough space
        rhos = np.arange(-diagonal, diagonal + 1, self.rho_resolution)

        # Generate an array of possible values for the theta parameter in the Hough space
        thetas = np.deg2rad(np.arange(0, 180, self.theta_resolution))

        # Create an empty Hough accumulator array
        H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

        # Get the indices of the non-zero pixels in the edge image
        y_idxs, x_idxs = np.nonzero(edge_image)

        # Vectorized implementation of the Hough Transform using broadcasting
        dot_products = x_idxs[:, np.newaxis] * np.cos(thetas) + y_idxs[:, np.newaxis] * np.sin(thetas)
        rho_idxs = np.round((dot_products + diagonal) / self.rho_resolution).astype(int)
        np.add.at(H, (rho_idxs, np.arange(len(thetas))), 1)

        # Return the Hough accumulator array and the arrays of possible rho and theta values
        return H, thetas, rhos

    def hough_peaks(self, H, thetas, rhos, threshold, num_peaks):
        # Initialize an empty list to store the peak values
        peaks = []

        # Make a copy of the Hough accumulator array
        H_copy = H.copy()

        # Loop over the specified number of peaks
        for _ in range(num_peaks):
            # Find the index of the maximum value in the Hough accumulator array
            idx = np.argmax(H_copy)

            # Convert the index to the corresponding (rho, theta) values
            rho_idx, theta_idx = np.unravel_index(idx, H_copy.shape)

            # Check if the value at the (rho, theta) bin exceeds the specified threshold
            if H_copy[rho_idx, theta_idx] >= threshold:
                # Add the (rho, theta) values to the list of peaks
                peaks.append((thetas[theta_idx], rhos[rho_idx]))

                # Set the value at the (rho, theta) bin to zero to prevent selecting the same peak again
                H_copy[rho_idx, theta_idx] = 0
            else:
                # If the value at the (rho, theta) bin is below the threshold, break out of the loop
                break

        # Return the list of peak values
        return peaks

    def hough_lines_draw(self, img, peaks):
        # Loop over all peak values
        for theta, rho in peaks:
            # Calculate the sine and cosine of the theta value
            a = np.cos(theta)
            b = np.sin(theta)

            # Calculate the (x, y) coordinates of two points on the line
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Draw a line on the input image using the calculated (x, y) coordinates
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Return the input image with the lines drawn on it
        return img

# Define the input and output directories
input_dir = "./input/"
output_dir = "./output/"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the input image paths
input_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".jpg")]
input_paths = sorted(input_paths, key=lambda x: os.path.basename(x))

# Define the Hough Transform parameters
threshold = 100
num_peaks = 100

# Loop over the input images
for i, input_path in enumerate(input_paths):
    # Load the input image
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    org_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    org_edges = cv2.Canny(gray, 50, 150)

    # Apply the Hough Transform using our implementation
    hough_transform = HoughTransform()
    H, thetas, rhos = hough_transform.hough_lines_acc(edges)
    peaks = hough_transform.hough_peaks(H, thetas, rhos, threshold=threshold, num_peaks=num_peaks)
    result_custom = hough_transform.hough_lines_draw(img, peaks)

    # Apply the Hough Transform using OpenCV
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=threshold)
    result_opencv = img.copy()
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(result_opencv, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Save the results to the output directory
    output_path_custom = os.path.join(output_dir, f"result_{i+1}_custom.jpg")
    output_path_opencv = os.path.join(output_dir, f"result_{i+1}_opencv.jpg")
    cv2.imwrite(output_path_custom, result_custom)
    cv2.imwrite(output_path_opencv, result_opencv)

    # Display the results side by side
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0][0].imshow(cv2.cvtColor(org_gray, cv2.COLOR_BGR2RGB))
    axs[0][0].set_title("Gray-Scale Image")
    axs[0][0].set_xlabel(f"Figure {i+1}.1")
    axs[0][0].set_xticks([])
    axs[0][0].set_yticks([])

    axs[0][1].imshow(cv2.cvtColor(org_edges, cv2.COLOR_BGR2RGB))
    axs[0][1].set_title("Canny Edge Output")
    axs[0][1].set_xlabel(f"Figure {i+1}.2")
    axs[0][1].set_xticks([])
    axs[0][1].set_yticks([])

    axs[1][0].imshow(cv2.cvtColor(result_custom, cv2.COLOR_BGR2RGB))
    axs[1][0].set_title("My Hough Transform")
    axs[1][0].set_xlabel(f"Figure {i+1}.3")
    axs[1][0].set_xticks([])
    axs[1][0].set_yticks([])

    axs[1][1].imshow(cv2.cvtColor(result_opencv, cv2.COLOR_BGR2RGB))
    axs[1][1].set_title("OpenCV Hough Transform")
    axs[1][1].set_xlabel(f"Figure {i+1}.4")
    axs[1][1].set_xticks([])
    axs[1][1].set_yticks([])
    
    plt.tight_layout()
    plt.show()


    output_path_matlab = os.path.join(output_dir, f"output_matplot_{i+1}.jpg")
    fig.savefig(output_path_matlab, bbox_inches='tight')