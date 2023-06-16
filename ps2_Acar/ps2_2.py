import cv2
import numpy as np
import time
import os

class HoughTransform:
    def __init__(self, theta_resolution=1, rho_resolution=1):
        self.theta_resolution = theta_resolution
        self.rho_resolution = rho_resolution
        self.thetas = np.deg2rad(np.arange(-90, 90, self.theta_resolution))
        self.cos_thetas = np.cos(self.thetas)
        self.sin_thetas = np.sin(self.thetas)

    def hough_lines_acc(self, edge_image):
        height, width = edge_image.shape
        max_distance = int(np.hypot(height, width))
        rhos = np.linspace(-max_distance, max_distance, max_distance * 2 // self.rho_resolution)

        accumulator = np.zeros((len(rhos), len(self.thetas)), dtype=np.uint64)

        y_idxs, x_idxs = np.nonzero(edge_image)

        # Vectorized implementation of the Hough Transform using broadcasting
        dot_products = x_idxs[:, np.newaxis] * self.cos_thetas + y_idxs[:, np.newaxis] * self.sin_thetas
        rho_idxs = np.round((dot_products + max_distance) / self.rho_resolution).astype(int)
        np.add.at(accumulator, (rho_idxs, np.arange(len(self.thetas))), 1)

        return accumulator, self.thetas, rhos


    def hough_peaks(self, H, thetas, rhos, threshold, num_peaks):
        peaks = []
        H_copy = H.copy()

        # Thresholding
        H_copy[H_copy < threshold] = 0

        for _ in range(num_peaks):
            idx = np.argmax(H_copy)
            rho_idx, theta_idx = np.unravel_index(idx, H_copy.shape)
            if H_copy[rho_idx, theta_idx] > 0:
                peaks.append((rhos[rho_idx], thetas[theta_idx]))
                H_copy[rho_idx, theta_idx] = 0
            else:
                break

        return peaks

    def hough_lines_draw(self, img, peaks, color=(0, 0, 255)):
        for rho, theta in peaks:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img, (x1, y1), (x2, y2), color, 2)

def main():
    cap = cv2.VideoCapture(0)

    # Set the resolution of the frames
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize the previous time and the start time
    prev_time = 0
    start_time = time.time()

    hough_transform = HoughTransform()

    # Define the output directory for the frames
    output_dir = "./output/video_frames"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 0

    while True:
        # Get the current time and compute the FPS
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time
        
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Downscale the frame to reduce the number of pixels to process(For More Performance)
        #frame = cv2.resize(frame, (320, 240))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines using the custom Hough Transform implementation
        H, thetas, rhos = hough_transform.hough_lines_acc(edges)
        peaks = hough_transform.hough_peaks(H, thetas, rhos, threshold=100, num_peaks=10)

        # Draw the lines on the frame
        hough_transform.hough_lines_draw(frame, peaks, color=(0, 0, 255))

        # Display the frame with a custom window name
        cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Hough Transform By Utku Acar', frame)

        # Save the frame as an image file
        output_filename = f"{output_dir}/frame_{frame_count:04d}.jpg"
        cv2.imwrite(output_filename, frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Stop recording after 15 seconds
        if time.time() - start_time >= 15:
            break

    cap.release()

    # Define the output video filename
    output_filename = "./output/output_video.avi"

    # Get the list of image files in the "./output/video_frames/" directory
    image_files = sorted([os.path.join("./output/video_frames/", f) for f in os.listdir("./output/video_frames/") if f.endswith(".jpg")])

    # Read the first image to get the frame size
    frame = cv2.imread(image_files[0])
    height, width, channels = frame.shape

    # Define the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, 15, (width, height))

    # Loop through the image files and add them to the video
    for image_file in image_files:
        frame = cv2.imread(image_file)
        out.write(frame)

    # Release the VideoWriter object and destroy all windows
    out.release()
    cv2.destroyAllWindows()
    
    # Delete the "./output/video_frames" directory and its contents
    import shutil
    shutil.rmtree(output_dir)

if __name__ == '__main__':
    main()