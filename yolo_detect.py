import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# --- Argument Parser ---
# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                     required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                     image folder ("test_dir"), video file ("testvid.mp4"), \
                     index of USB camera ("usb0"), or RTSP stream URL ("rtsp://user:pass@ip:port/path")',
                     required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                     default=0.5, type=float) # Added type=float
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                     otherwise, match source resolution',
                     default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                     action='store_true')

args = parser.parse_args()


# --- Parse User Inputs ---
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# --- Determine Source Type ---
# Parse input to determine if image source is a file, folder, video, USB camera, or RTSP stream
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif img_source.startswith('usb'): # Use .startswith for more robustness
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif img_source.startswith('picamera'): # Use .startswith for more robustness
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
elif img_source.startswith('rtsp://'): # --- ADDED: RTSP Stream Check ---
    source_type = 'rtsp'
else:
    print(f'Input {img_source} is invalid. Please try again. Supported types: file, folder, video, usbX, picameraX, rtsp://URL')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    try:
        resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
    except ValueError: # Handle potential error if resolution format is wrong
        print(f"Error: Invalid resolution format '{user_res}'. Please use WxH (e.g., 1280x720).")
        sys.exit(0)

# Check if recording is valid and set up recording
if record:
    # --- UPDATED: Allow RTSP for recording ---
    if source_type not in ['video', 'usb', 'rtsp', 'picamera']: # Added rtsp and picamera
        print('Recording only works for video, camera, or RTSP sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)

    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30 # You might want to get this from the source if possible
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))


# --- Load or Initialize Image Source ---
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(os.path.join(img_source, '*')) # Use os.path.join for robustness
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb' or source_type == 'rtsp': # --- ADDED: RTSP Source ---
    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    elif source_type == 'rtsp': cap_arg = img_source # Use the RTSP URL directly

    cap = cv2.VideoCapture(cap_arg)

    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"ERROR: Could not open video source {cap_arg}. Please check the source path/index/URL.")
        sys.exit(0)

    # Set camera or video resolution if specified by user
    if user_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    # Ensure resolution is set for picamera
    if not user_res:
        print("ERROR: Resolution must be specified for Picamera2 source.")
        sys.exit(0)
    # picamera2 typically takes (width, height) directly
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# --- Begin Inference Loop ---
while True:
    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder': # If source is image or image folder, load the image using its filename
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1

    # --- UPDATED: Handle video, USB, and RTSP uniformly ---
    elif source_type in ['video', 'usb', 'rtsp']:
        ret, frame = cap.read()
        if not ret:
            print('Unable to read frames from the source. This indicates the source is disconnected or not working. Exiting program.')
            break

    elif source_type == 'picamera': # If source is a Picamera, grab frames using picamera interface
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if (frame is None): # Check after conversion
            print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    # Resize frame to desired display resolution
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))

    # Run inference on frame
    results = model(frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    # Initialize variable for basic object counting example
    object_count = 0

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):
        # Get bounding box coordinates
        # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
        xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get bounding box confidence
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > min_thresh: # Use min_thresh from arguments
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

            # Basic example: count the number of objects in the image
            object_count = object_count + 1

    # Calculate and draw framerate (if using video, USB, Picamera, or RTSP source)
    if source_type in ['video', 'usb', 'picamera', 'rtsp']: # --- UPDATED: FPS for RTSP ---
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw framerate

    # Display detection results
    cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw total number of detected objects
    cv2.imshow('YOLO detection results',frame) # Display image
    if record: recorder.write(frame)

    # If inferencing on individual images, wait for user keypress before moving to next image. Otherwise, wait 5ms before moving to next frame.
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type in ['video', 'usb', 'picamera', 'rtsp']: # --- UPDATED: waitKey for RTSP ---
        key = cv2.waitKey(5)

    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png',frame)

    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS for past frames
    avg_frame_rate = np.mean(frame_rate_buffer)


# --- Clean up ---
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
# --- UPDATED: Release cap for RTSP ---
if source_type in ['video', 'usb', 'rtsp']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()