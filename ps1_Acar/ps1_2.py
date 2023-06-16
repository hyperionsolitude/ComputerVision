import cv2
import numpy as np
import time

# Create a VideoCapture object to capture video from the default webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is successfully opened
if not cap.isOpened():
    print('Unable to open the webcam')
    exit()

# Get the frame width and height of the video stream
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define Sobel filters for x and y directions
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Define the font and position to display FPS on the video frames
font = cv2.FONT_HERSHEY_SIMPLEX
fps_pos = (10, 30)

# Initialize the previous time and the start time
prev_time = 0
start_time = time.time()

# Create a VideoWriter object to record the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./output/ps1_2_output.avi', fourcc, 30.0, (frame_width * 2, frame_height))

# Start reading and processing the frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # If there is an error reading the frame, exit the loop
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute the gradient magnitude using Sobel filters
    gradient_x = cv2.filter2D(gray_frame, -1, sobel_x).astype(np.float32)
    gradient_y = cv2.filter2D(gray_frame, -1, sobel_y).astype(np.float32)
    
    #np.clip to limit the gradient magnitude to [0,255] range
    gradient_magnitude = np.clip(cv2.magnitude(gradient_x, gradient_y), 0, 255).astype(np.uint8)
    
    # Get the current time and compute the FPS
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time))
    prev_time = curr_time

    # Resize the input frame to match the size of the gradient magnitude image
    resized_frame = cv2.resize(frame, (gradient_magnitude.shape[1], gradient_magnitude.shape[0]))

    # Concatenate the input frame and the gradient magnitude image side by side
    combined_frame = np.concatenate((resized_frame, cv2.cvtColor(gradient_magnitude, cv2.COLOR_GRAY2BGR)), axis=1)

    # Display the combined frame on the screen and write it to the output video
    cv2.putText(combined_frame, f'FPS: {fps}', fps_pos, font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Combined', combined_frame)
    out.write(combined_frame.astype(np.uint8))

    # Wait for the 'q' key to be pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Stop recording after 15 seconds
    if time.time() - start_time >= 15:
        break

# Release the VideoCapture and VideoWriter objects and destroy all windows
cap.release()
out.release()
cv2.destroyAllWindows()
