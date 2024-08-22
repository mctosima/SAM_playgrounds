import re
import os
import numpy as np
import cv2
from glob import glob
import sys
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

import cv2
import numpy as np
import mediapipe as mp

def detect_three_roi(img_frame_seq):
    '''
    Input:
        - img_frame_seq (list of numpy.ndarray): list of frames from a video
        - Input in RGB format
    '''
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5)
    
    # Interest points (left cheek, right cheek, forehead, tracking point)
    interest_points = [205, 425, 151, 4]
    
    # Get the facial landmark for the first frame
    results = face_mesh.process(img_frame_seq[0])
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    
    # Get the x, y, z coordinates of the interest points
    height, width, _ = img_frame_seq[0].shape
    int_point = [
        (int(landmarks[interest_points[i]].x * width), int(landmarks[interest_points[i]].y * height))
        for i in range(4)
    ]
    
    # Define the initial ROIs
    roi_size_square = 28
    roi_size_rect = (70, 30)
    roi_track_point_size = (45, 45)
    
    roi = []
    roi.append([(int_point[0][0]-roi_size_square//2, int_point[0][1]-roi_size_square//2), 
                (int_point[0][0]+roi_size_square//2, int_point[0][1]+roi_size_square//2)])
    roi.append([(int_point[1][0]-roi_size_square//2, int_point[1][1]-roi_size_square//2), 
                (int_point[1][0]+roi_size_square//2, int_point[1][1]+roi_size_square//2)])
    roi.append([(int_point[2][0]-roi_size_rect[0]//2, int_point[2][1]-roi_size_rect[1]//2), 
                (int_point[2][0]+roi_size_rect[0]//2, int_point[2][1]+roi_size_rect[1]//2)])
    roi.append([(int_point[3][0]-roi_track_point_size[0]//2, int_point[3][1]-roi_track_point_size[1]//2), 
                (int_point[3][0]+roi_track_point_size[0]//2, int_point[3][1]+roi_track_point_size[1]//2)])
    
    # Manually adjust the tracking ROI. Shift the y2
    y2_shift = 40
    roi[3][1] = (roi[3][0][0]+roi_track_point_size[0], roi[3][0][1]+roi_track_point_size[1]+y2_shift)
    
    # Initialize the list to store ROIs for all frames
    rois_over_time = []

    # Convert first frame to grayscale for optical flow
    prev_gray = cv2.cvtColor(img_frame_seq[0], cv2.COLOR_RGB2GRAY)
    
    # Store the initial ROIs
    initial_rois = np.array(roi).reshape(-1, 4)
    rois_over_time.append(initial_rois)
    
    # Lucas-Kanade parameters
    lk_params = dict(winSize=(12, 12), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    for i in range(1, len(img_frame_seq)):
        frame_gray = cv2.cvtColor(img_frame_seq[i], cv2.COLOR_RGB2GRAY)

        # Track the points in the fourth ROI
        p0 = np.array([[roi[3][0]], [roi[3][1]]], dtype=np.float32).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)

        if st[0] == 1:
            # Update the tracking point
            dx, dy = p1[0, 0] - p0[0, 0]
            for j in range(4):
                roi[j] = [(roi[j][0][0] + dx, roi[j][0][1] + dy),
                          (roi[j][1][0] + dx, roi[j][1][1] + dy)]

            # Check if the movement is drastic (e.g., large displacement)
            displacement_treshold = 10
            if np.linalg.norm([dx, dy]) > displacement_treshold:  # Threshold for re-detection
                results = face_mesh.process(img_frame_seq[i])
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    int_point = [
                        (int(landmarks[interest_points[k]].x * width), int(landmarks[interest_points[k]].y * height))
                        for k in range(4)
                    ]
                    # Redefine the ROIs
                    roi[0] = [(int_point[0][0]-roi_size_square//2, int_point[0][1]-roi_size_square//2), 
                              (int_point[0][0]+roi_size_square//2, int_point[0][1]+roi_size_square//2)]
                    roi[1] = [(int_point[1][0]-roi_size_square//2, int_point[1][1]-roi_size_square//2), 
                              (int_point[1][0]+roi_size_square//2, int_point[1][1]+roi_size_square//2)]
                    roi[2] = [(int_point[2][0]-roi_size_rect[0]//2, int_point[2][1]-roi_size_rect[1]//2), 
                              (int_point[2][0]+roi_size_rect[0]//2, int_point[2][1]+roi_size_rect[1]//2)]
                    roi[3] = [(int_point[3][0]-roi_track_point_size[0]//2, int_point[3][1]-roi_track_point_size[1]//2), 
                              (int_point[3][0]+roi_track_point_size[0]//2, int_point[3][1]+roi_track_point_size[1]//2)]
                    roi[3][1] = (roi[3][0][0]+roi_track_point_size[0], roi[3][0][1]+roi_track_point_size[1]+y2_shift)

        # Store the current ROIs
        current_rois = np.array(roi).reshape(-1, 4)
        rois_over_time.append(current_rois)
        
        # Update the previous frame
        prev_gray = frame_gray.copy()

    # Convert list to numpy array
    roi_np_array = np.array(rois_over_time)
    
    # Apply sliding moving average with a window size of 5
    smoothed_roi_np_array = np.zeros_like(roi_np_array)
    window_size = 5

    for i in range(len(roi_np_array)):
        if i < window_size:
            smoothed_roi_np_array[i] = np.mean(roi_np_array[:i+1], axis=0)
        else:
            smoothed_roi_np_array[i] = np.mean(roi_np_array[i-window_size+1:i+1], axis=0)


    return smoothed_roi_np_array

def create_roi_preview_video(roi_np_array, img_frame_seq, output_path='./out/roi_preview.mp4', fps=30):
    '''
    Creates a preview video showing the ROIs on each frame.
    
    Input:
        - roi_np_array: NumPy array of shape (frames, ROI-cat, the_ROI)
        - img_frame_seq: list of numpy.ndarray, list of frames from a video
        - output_path: string, path to save the output video file
        - fps: int, frames per second for the output video
    '''
    
    # Get the frame dimensions
    height, width, _ = img_frame_seq[0].shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For mp4 output
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Loop through each frame and draw the ROIs
    for i in range(len(img_frame_seq)):
        frame = img_frame_seq[i].copy()  # Copy the frame to draw on it
        
        # Get the ROIs for the current frame
        rois = roi_np_array[i]
        
        # Draw the ROIs
        cv2.rectangle(frame, (int(rois[0][0]), int(rois[0][1])), (int(rois[0][2]), int(rois[0][3])), (255, 0, 0), 2)  # First ROI
        cv2.rectangle(frame, (int(rois[1][0]), int(rois[1][1])), (int(rois[1][2]), int(rois[1][3])), (255, 0, 0), 2)  # Second ROI
        cv2.rectangle(frame, (int(rois[2][0]), int(rois[2][1])), (int(rois[2][2]), int(rois[2][3])), (0, 255, 0), 2)  # Third ROI (rectangular)
        cv2.rectangle(frame, (int(rois[3][0]), int(rois[3][1])), (int(rois[3][2]), int(rois[3][3])), (0, 0, 255), 2)  # Tracking point ROI
        
        # Write the frame into the video file
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Release the VideoWriter object
    video_writer.release()
    # print(f"Video saved as {output_path}")
    
def extract_avg_pixel_value(img_frames, three_rois):
    '''
    Extract the average pixel value from every frame for each of the first three ROIs and each RGB channel.
    
    Input:
        - img_frames: list of numpy.ndarray, list of frames from a video in RGB format.
        - three_rois: numpy.ndarray, array containing ROIs with shape [temporal, 4 (ROI), 4 (x1, y1, x2, y2)].
        
    Output:
        - avg_pixel_values: numpy.ndarray, array of average pixel values with shape [temporal, 3 (ROI), 3 (RGB channels)].
    '''
    
    num_frames = len(img_frames)
    num_rois = 3  # We are only interested in the first 3 ROIs
    num_channels = 3  # RGB channels
    
    # Initialize the array to store the average pixel values
    avg_pixel_values = np.zeros((num_frames, num_rois, num_channels))
    
    # Iterate over each frame
    for i in range(num_frames):
        frame = img_frames[i]
        
        # Iterate over the first three ROIs
        for j in range(num_rois):
            x1, y1, x2, y2 = map(int, three_rois[i, j])
            roi_pixels = frame[y1:y2, x1:x2, :]
            
            # Calculate the mean pixel value for each channel
            avg_pixel_values[i, j] = np.mean(roi_pixels, axis=(0, 1))
    
    return avg_pixel_values


if __name__ == '__main__':
    PURE_PATH = '/mnt/disk2/PURE'
    FPS = 30
    OUTPUT_DIR = './threeROI_PURE'
    ERROR_LOG_PATH = os.path.join(OUTPUT_DIR, 'error_log.txt')
    
    # Clear the previous error log (if exists)
    if os.path.exists(ERROR_LOG_PATH):
        os.remove(ERROR_LOG_PATH)
    
    # Populate subdirectories
    subject = [
        name for name in os.listdir(PURE_PATH)
        if os.path.isdir(os.path.join(PURE_PATH, name)) and re.search(r'\d$', name)
    ]
    subject.sort()
    
    # Start from a specific subject
    start_from = 0
    subject = subject[start_from:]
    print(f"Subjects: {subject}")
    print(f"Total subjects: {len(subject)}")
    
    for i, sub in enumerate(subject):
        print(f"Processing {sub} ({i+1}/{len(subject)})")
        directory = os.path.join(PURE_PATH, sub)
        print(f"Path: {directory}")
        
        try:
            # Get all the frames of .jpg or .png format and convert to numpy array
            img_frames = []
            ext_type = ['jpg', 'png', 'jpeg']
            all_imgs_path = []
            for ext in ext_type:
                all_imgs_path += glob(os.path.join(directory, f'*.{ext}'))
            
            # Sort the images
            all_imgs_path.sort()
        
            for file in all_imgs_path:
                img = cv2.imread(file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_frames.append(img)
            
            img_frames = np.array(img_frames)
            
            # Get the three ROI
            three_rois = detect_three_roi(img_frames)
            if three_rois is None:
                error_message = f"Error: No landmarks detected for {sub}. Skipping...\n"
                print(error_message)
                with open(ERROR_LOG_PATH, 'a') as f:
                    f.write(error_message)
                continue
            
            # Preview the three ROI
            vid_filename = os.path.join(OUTPUT_DIR, "roi_preview", f'{sub}.mp4')
            create_roi_preview_video(three_rois, img_frames, output_path=vid_filename, fps=FPS)
            
            # Extract the average pixel value
            avg_pixel_values = extract_avg_pixel_value(img_frames, three_rois)
            
            # Save the average pixel value (shape format: [temporal, 3 (ROI), 3 (RGB channels)])
            np.save(os.path.join(OUTPUT_DIR, "avg_pixel_value", f'{sub}.npy'), avg_pixel_values)
        
        except Exception as e:
            error_message = f"Error processing {sub}: {e}\n"
            print(error_message)
            with open(ERROR_LOG_PATH, 'a') as f:
                f.write(error_message)
            continue
            
        # break # break for subject loop (remove or uncomment for testing)
