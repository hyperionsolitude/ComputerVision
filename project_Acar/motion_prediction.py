#####################################Model & Code Citations####################################
#                                                                                             #
# https://docs.opencv.org/4.7.0/                                                              #
# https://github.com/AlexeyAB/darknet/raw/master/cfg/yolov4.cfg                               #
# https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights#
#                                                                                             #
###############################################################################################
import cv2
import numpy as np
import time
import sys

# Initialize variables for highest velocity frame
fastest_motion_frame = None
fastest_motion = 0.0
fastest_frame = 0
prev_p1 = None
prev_p2 = None
# Initialize the input arguments
option = 0
secop = 1

num_args = len(sys.argv)

if num_args > 1:
    option = int(sys.argv[1])

    if option == 0:
        cfg = "./cfg/yolov4-tiny.cfg"
        weights = "./weights/yolov4-tiny.weights"
        output_video_name = './output/yolov4-tiny_camera_output'
        frame_title = "yolov4-tiny Tracking & Trajectory Prediction"
        input_path = 0
    
    if option == 1:
        cfg = "./cfg/yolov4-tiny.cfg"
        weights = "./weights/yolov4-tiny.weights"
        output_video_name = './output/yolov4-tiny_video_output'
        frame_title = "yolov4-tiny Tracking & Trajectory Prediction"
        input_path = "./input/camera.mp4"
    
    if option == 2:
        cfg = "./cfg/yolov4.cfg"
        weights = "./weights/yolov4.weights"
        output_video_name = './output/yolov4_camera_output'
        frame_title = "yolov4 Tracking & Trajectory Prediction"
        input_path = 0

    if option == 3:
        cfg = "./cfg/yolov4.cfg"
        weights = "./weights/yolov4.weights"
        output_video_name = './output/yolov4_video_output'
        frame_title = "yolov4-tiny Tracking & Trajectory Prediction"
        input_path = "./input/camera.mp4"

    if option == 4:
        cascade_xml = "./cfg/haarcascade_frontalface_default.xml"
        output_video_name = './output/haarcascade_video_output'
        frame_title = "Haar Cascade Face Detection & Tracking"
        input_path = 0

    if option == 5:
        cascade_xml = "./cfg/haarcascade_frontalface_default.xml"
        output_video_name = './output/haarcascade_video_output'
        frame_title = "Haar Cascade Face Detection & Tracking"
        input_path = "./input/camera.mp4"

# Load YOLOv4 model if option is not 4
if option < 4:
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    output_layers = net.getUnconnectedOutLayersNames()

# Initialize HOG descriptor
hog = cv2.HOGDescriptor()

# Object detection using YOLOv4
def detect_objects(image, net):
    # Create a blob from the input image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Forward pass through the network
    layer_outputs = net.forward(output_layers)

    # Get the bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x, center_y, width, height = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                x, y = int(center_x - width / 2), int(center_y - height / 2)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.4)

    # Return the bounding boxes of the detected objects
    objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            objects.append((x, y, x + w, y + h))
    else:
        print("The model didn't find any valid object.")

    return objects

# Extract features using HOG descriptor
def extract_features(image, objects):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the HOG features for each object
    features = []
    hog = cv2.HOGDescriptor()
    for obj in objects:
        x1, y1, x2, y2 = obj
        roi = gray[y1:y2, x1:x2]
        if roi.size != 0:  # Check if ROI has non-empty size
            roi = cv2.resize(roi, (64, 128))
            feature = hog.compute(roi)
            features.append(feature)

    return features

# MeanShift object tracking
def track_objects(image, objects, features, prev_objects=None, dt=1.0):
    # Initialize the tracker
    tracker = cv2.TrackerKCF_create()

    # Track the objects in the current frame
    for i, obj in enumerate(objects):
        tracker.init(image, obj)

        # Update the tracker for each frame
        ok, bbox = tracker.update(image)
        if ok:
            x, y, w, h = bbox
            objects[i] = (int(x), int(y), int(x + w), int(y + h))

            if prev_objects is not None:
                # Calculate velocity of the object
                prev_obj = prev_objects[i]
                displacement = np.array(obj) - np.array(prev_obj)
                velocity = displacement / dt
                objects[i] = objects[i] + velocity.astype(int)

    return objects

# Calculate IoU (Intersection over Union) between two bounding boxes
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Handle zero/zero division
    if box1_area + box2_area - intersection_area == 0:
        return 0.0
    
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


# Initialize the video capture
cap = cv2.VideoCapture(input_path)

# Get the first frame of the video
ret, frame = cap.read()

frame_counter = 1

# Detect objects in the first frame
if option < 4:
    objects = detect_objects(frame, net)
else:
    cascade = cv2.CascadeClassifier(cascade_xml)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Extract features for object representation
features = extract_features(frame, objects)

# Initialize tracker
tracker = cv2.TrackerCSRT_create()

# Initialize bounding box for tracking
if num_args > 2:
    secop = int(sys.argv[2])
    if secop == 0:
        bbox = cv2.selectROI(frame, False)
        output_video_name = output_video_name + "_manual_ROI"
    if secop == 1:
        if option != 4 and option != 5 and len(objects) > 0:
            bbox = objects[0]
        else:
            bbox = tuple(objects[0])
        output_video_name = output_video_name + "_auto_ROI"

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(output_video_name + ".mp4", fourcc, 30.0, (640, 480))

ok = tracker.init(frame, bbox)
prev_bbox = bbox
prev_time = time.time()

# Initialize FPS calculation variables
frame_count = 0
start_time = time.time()

# Initialize variables for IoU calculation
iou_values = []
iou_sum = 0.0

time_record_start = time.time()

# Loop through video frames
while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_counter = frame_counter + 1
    # Calculate time difference between previous and current frame
    curr_time = time.time()
    dt = curr_time - prev_time
    prev_time = curr_time

    # Update tracker
    ok, bbox = tracker.update(frame)

    # If tracking success, draw the tracked object and its predicted next frame
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)

        # Predict next frame of the object
        if prev_bbox is not None:
            displacement = np.array(p1) - np.array(prev_bbox[:2])
            velocity = displacement / dt
            next_p1 = tuple((np.array(p1) + velocity * dt).astype(int))
            next_p2 = tuple((np.array(p2) + velocity * dt).astype(int))

            # Draw the predicted next frame
            cv2.rectangle(frame, next_p1, next_p2, (0, 0, 255), 2, 1)

            # Draw arrows from corners of tracking frame to predicted frame corners
            cv2.arrowedLine(frame, p1, next_p1, (255, 0, 0), 2)
            cv2.arrowedLine(frame, (p1[0], p2[1]), (next_p1[0], next_p2[1]), (255, 0, 0), 2)
            cv2.arrowedLine(frame, (p2[0], p1[1]), (next_p2[0], next_p1[1]), (255, 0, 0), 2)
            cv2.arrowedLine(frame, p2, next_p2, (255, 0, 0), 2)
            
            # Calculate velocity magnitude
            motion = np.linalg.norm(velocity * dt)
            
            # Check if current velocity is the highest
            if motion > fastest_motion:
                fastest_motion = motion
                fastest_motion_frame = frame.copy()
                fastest_frame = frame_counter
            
            # Calculate IoU between current tracking box and previous predicted box
            if prev_p1 and prev_p2 != None:
                #print("Prev_p1:",prev_p1,"Prev_p2:",prev_p2)
                iou = calculate_iou((prev_p1[0], prev_p1[1], prev_p2[0] - prev_p1[0], prev_p2[1] - prev_p1[1]), bbox)
                iou_values.append(iou)
                iou_sum += iou
            #print("Next_p1:",next_p1,"Next_p2:",next_p2)
            prev_p1 = (int(next_p1[0]), int(next_p1[1]))
            prev_p2 = (int(next_p2[0]), int(next_p2[1]))

        prev_bbox = bbox

    # Calculate and display FPS
    frame_count += 1
    fps = frame_count / (curr_time - start_time)
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    frame_count = 0
    start_time = curr_time

    # Display the resulting frame
    cv2.imshow(frame_title, frame)

    # Write the frame into the output video
    output.write(frame)
    
    elapsed_time = time.time() - time_record_start
    
    if input_path == 0 and elapsed_time >= 15:
         break
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if fastest_motion_frame is not None:
    fastest_motion_frame_name = "fastest_motion_frame_" + output_video_name + ".jpg"
    cv2.imwrite(fastest_motion_frame_name, fastest_motion_frame)

# Release the video capture and writer
cap.release()
output.release()

# Calculate average IoU
if len(iou_values) > 0:
    avg_iou = iou_sum / len(iou_values)
    print("Average IoU:", avg_iou)
    
# Write average FPS and average IoU to a text file
tag = f"option{output_video_name}"
output_file = f"./output/results.txt"
with open(output_file, 'a') as f:
    f.write(f"\nConfig --> {output_video_name}:\n")
    f.write(f"Average FPS: {fps}\n")
    f.write(f"Average IoU: {avg_iou}\n")
    if fastest_motion_frame is not None:
        f.write(f"Fastest Motion: {fastest_motion} @:{fastest_frame} --> {fastest_motion_frame_name}\n")
# Close all windows
cv2.destroyAllWindows()
