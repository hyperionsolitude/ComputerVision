import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculateHomography(correspondences):
    # Convert correspondences to numpy arrays
    src_points = np.float32([correspondence[0] for correspondence in correspondences])
    dst_points = np.float32([correspondence[1] for correspondence in correspondences])

    # Calculate the homography matrix using RANSAC
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    return H

def warpImage(img, H):
    # Warp the image using the homography matrix
    warped_img = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))

    return warped_img

def get_stitched_image(img1, img2, H):
    # Warp the second image to align with the first image
    img2_warped = warpImage(img2, H)

    # Stitch the images together
    result = cv2.addWeighted(img1, 0.5, img2_warped, 0.5, 0)

    return result

METHODS = {'SIFT', 'ORB', 'SURF'}

# Read the images
img1 = cv2.imread('input/img1.ppm')
img2 = cv2.imread('input/img2.ppm')

cv2.imwrite("input/img1.png", img1)
cv2.imwrite("input/img2.png", img2)
# Convert the images to grayscale
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

for METHOD in METHODS:
    print()
    if METHOD == 'SIFT':
        
        print('Calculating SIFT features...\n')
        
        
        # Create a SIFT object
        sift = cv2.xfeatures2d.SIFT_create()

        # Get the keypoints and the descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)
        # keypoints object includes position, size, angle, etc.
        # descriptors is an array. For sift, each row is a 128-length feature vector
        print('Descriptor 1:', len(descriptors1))
        print('Descriptor 2:', len(descriptors2))
    elif METHOD == 'SURF':
        print('Calculating SURF features...\n')
        
        # Create a SURF object
        surf = cv2.xfeatures2d.SURF_create()
        keypoints1, descriptors1 = surf.detectAndCompute(img1_gray, None)
        keypoints2, descriptors2 = surf.detectAndCompute(img2_gray, None)
        print('Descriptor 1:', len(descriptors1))
        print('Descriptor 2:', len(descriptors2))
    elif METHOD == 'ORB':
        print('Calculating ORB features...\n')
        # Create an ORB object
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)
        print('Descriptor 1:', len(descriptors1))
        print('Descriptor 2:', len(descriptors2))
        # Note: Try cv2.NORM_HAMMING for this feature
        

    # Draw the keypoints
    img1_key = np.zeros((img1.shape))
    img1_key = cv2.drawKeypoints(image=img1, outImage=img1_key, keypoints=keypoints1,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                 color=(0, 0, 255))
    img2_key = np.zeros((img2.shape))
    img2_key = cv2.drawKeypoints(image=img2, outImage=img2_key, keypoints=keypoints2,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                 color=(0, 0, 255))

    # Display the images
    cv2.imwrite(f"output/{METHOD} Keypoints 1.png", img1_key)
    cv2.imwrite(f"output/{METHOD} Keypoints 2.png", img2_key)

    # Create a brute-force descriptor matcher object
    if METHOD == 'ORB':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    matching = bf.knnMatch(descriptors1, descriptors2, k=2)

    print("# of matches between descriptors:", len(matching))

    # ------------------------------------------------------------------------
    # Problem-1
    # Best Match
    bestMatches = []
    for m, n in matching:
        if m.distance < 0.75 * n.distance:
            bestMatches.append(m)

    print("# of best matches:", len(bestMatches))

    # Save the best matches visualization
    best_matches_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, bestMatches, None,
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(f"output/{METHOD}_Matches.png", best_matches_img)
            

    # ------------------------------------------------------------------------
    # Repeatability
    matches = bf.match(descriptors1, descriptors2)

    # Apply the given homography to obtain ground truth points
    h_mat_ground_truth = np.loadtxt('./input/H1to2p')

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2_ground_truth = cv2.perspectiveTransform(points1, h_mat_ground_truth)

    # Compute the number of correct matches
    correct_matches = 0
    for i, match in enumerate(matches):
        point2 = keypoints2[match.trainIdx].pt
        point2_ground_truth = points2_ground_truth[i, 0]
        distance = np.linalg.norm(point2 - point2_ground_truth)
        if distance <= 5.0:  # Set a threshold for correctness
            correct_matches += 1

    # Calculate the average number of points extracted in the input images
    avg_points = (len(keypoints1) + len(keypoints2)) / 2

    # Calculate the repeatability rate
    repeatability_rate = correct_matches / avg_points

    print("Repeatability Rate:", repeatability_rate)
    # -------------------------------------------------------------------------
    # Homography Estimation
    correspondences = [(keypoints1[m.queryIdx].pt, keypoints2[m.trainIdx].pt) for m in matches]
    estimatedH = calculateHomography(correspondences)

    print("Given homography matrix:")
    print(h_mat_ground_truth)
    print("Estimated homography matrix:")
    print(estimatedH)

    #-----------------------------------------------------------------------------
    # Warp and display the residual image
    warped_img = warpImage(img1, estimatedH)
    residual = cv2.absdiff(img2, warped_img)
    
    # Show the Residual Image
    residual_rgb = cv2.cvtColor(residual, cv2.COLOR_BGR2RGB)
    plt.imshow(residual_rgb)
    plt.axis('off')
    plt.title(f"Residual Image by {METHOD}")
    plt.show()
    cv2.imwrite(f"output/{METHOD}_Residual.png", residual)
    #-----------------------------------------------------------------------------
    # Image Stitching
    stitched_img = get_stitched_image(img1, img2, estimatedH)
    stitched_img_rgb = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB)
    plt.imshow(stitched_img_rgb)
    plt.axis('off')
    plt.title(f"Stitched Image by {METHOD}")
    plt.show()
    cv2.imwrite(f"output/{METHOD}_Stitched.png", stitched_img)

