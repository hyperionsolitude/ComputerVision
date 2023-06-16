import cv2
import time

# Define the video capture object
cap = cv2.VideoCapture(0)  # 0 represents the default camera

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter('camera.mp4', fourcc, 20.0, (640, 480))  # Output file name, codec, FPS, and frame size

fps = cap.get(cv2.CAP_PROP_FPS)
print("Webcam FPS:", fps)

# Set the duration of recording (in seconds)
record_duration = 15
start_time = time.time()
end_time = start_time + record_duration

# Start recording
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Write the captured frame into the output video file
    out.write(frame)

    cv2.imshow('Video Recording', frame)

    # Check if the recording duration has exceeded the specified time
    if time.time() > end_time:
        break

    # Check for 'q' key press to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
