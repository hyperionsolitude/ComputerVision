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

image_paths = sorted(image_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0][5:]))

# Set the size of the checkerboard squares in mm
square_size = 30

# Prepare object points (coordinates of the checkerboard corners in the real world)
pattern_size1 = (13, 13)  # Number of inner corners in the checkerboard pattern
pattern_size2 = (12, 12)  # Number of inner corners in the checkerboard pattern
pattern_size3 = (12, 13)  # Number of inner corners in the checkerboard pattern
pattern_size4 = (13, 12)  # Number of inner corners in the checkerboard pattern
objp = np.zeros((np.prod(pattern_size1), 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size1[0], 0:pattern_size1[1]].T.reshape(-1, 2)
objp *= square_size
obj_points = []  # 3D points of the checkerboard corners in the real world
img_points = []  # 2D points of the checkerboard corners in the images

for path in image_paths:
    # Load the image
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size1, None)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size2, None)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size3, None)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size4, None)
    print(path,",",ret,"",corners,"\n")
    if ret:
        obj_points.append(objp)
        img_points.apretpend(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(0)  # Display the image for a short duration
exit()
cv2.destroyAllWindows()

# Calibrate the camera and undistort the images
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
undistorted_images = []

for i, path in enumerate(image_paths):
    img = cv2.imread(path)
    undistorted_img = cv2.undistort(img, mtx, dist)
    undistorted_images.append(undistorted_img)

    # Save the undistorted image
    output_path = f"./output/image_{i+1}_undistorted.tif"
    cv2.imwrite(output_path, undistorted_img)


# Step (b): Estimate the intrinsic matrix
intrinsic_matrix = mtx
print("Intrinsic matrix:")
print(intrinsic_matrix)

# Save the intrinsic matrix to results.txt
output_file = 'results.txt'
with open(output_file, 'a') as f:
    f.write("Intrinsic matrix:\n")
    np.savetxt(f, intrinsic_matrix, delimiter=', ')
    f.write('\n\n')


# Step (c): Estimate the projection matrix for each image
projection_matrices = []

for i in range(len(image_paths)):
    rvec, _ = cv2.Rodrigues(rvecs[i])
    tvec = tvecs[i]
    projection_matrix = np.hstack((rvec, tvec))
    projection_matrices.append(projection_matrix)

    # Save the projection matrix to results.txt
    with open(output_file, 'a') as f:
        f.write(f"Projection matrix for image {i+1}:\n")
        np.savetxt(f, projection_matrix, delimiter=', ')
        f.write('\n\n')


# Step (d): Estimate the essential matrix between the first image and each of the remaining images
essential_matrices = []

for i in range(1, len(projection_matrices)):
    P1 = projection_matrices[0]
    P2 = projection_matrices[i]
    essential_matrix, _ = cv2.findEssentialMat(img_points[0], img_points[i], mtx)
    essential_matrices.append(essential_matrix)

    # Save the essential matrix to results.txt
    with open(output_file, 'a') as f:
        f.write(f"Essential matrix between image 1 and {i+1}:\n")
        np.savetxt(f, essential_matrix, delimiter=', ')
        f.write('\n\n')


# Step (e): Estimate the rotation and translation between the reference image and each of the remaining images
rotation_matrices = []
translation_vectors = []

for i in range(len(essential_matrices)):
    _, R, t, _ = cv2.recoverPose(essential_matrices[i], img_points[0], img_points[i], mtx)
    rotation_matrices.append(R)
    translation_vectors.append(t)

    # Save the rotation and translation to results.txt
    with open(output_file, 'a') as f:
        f.write(f"Rotation matrix for image {i+1}:\n")
        np.savetxt(f, R, delimiter=', ')
        f.write(f"Translation vector for image {i+1}:\n")
        np.savetxt(f, t, delimiter=', ')
        f.write('\n\n')
