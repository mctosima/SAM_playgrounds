import random
import os
import re
import numpy as np
import cv2
import traceback
from facialextractor import FacialExtractor
from glob import glob

PURE_PATH = '/mnt/disk2/PURE'
FPS = 30
OUTPUT_DIR = './masked_PURE'
REPORT_FILE = 'processing_report.txt'

def create_video_from_numpy(frames, output_path, fps=30):
    """
    Create an MP4 video from a numpy array of frames.

    Args:
        frames (numpy.ndarray): 4D array of frames (num_frames x height x width x channels)
        output_path (str): Path where the output video will be saved
        fps (int): Frames per second for the output video

    Returns:
        None
    """
    # Get the dimensions of the frames
    num_frames, height, width, channels = frames.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        out.write(frames[i])

    # Release the VideoWriter
    out.release()

def process_subject(sub_name, facial_extractor):
    """
    Process a single subject's video.

    Args:
        sub_name (str): Name of the subject directory
        facial_extractor (FacialExtractor): Instance of FacialExtractor

    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        print(f"Processing {sub_name}")
        img_path = os.path.join(PURE_PATH, sub_name)
        all_face_skins = facial_extractor.process_img_frames(
            image_path=img_path,
            use_klt=True,
            dlib_interval=0.5,
            landmark_interval=0.2,
            segment_interval=0.2,
            winSize=(20, 20),
            maxLevel=1,
            minDistance=7,
            maxCorners=75,
            smoothing_window=5,
            verbose=False,
        )

        # save the masked face skins
        np.save(os.path.join(OUTPUT_DIR, f'{sub_name}.npy'), all_face_skins)

        # save the video
        create_video_from_numpy(all_face_skins, os.path.join(OUTPUT_DIR, f'{sub_name}.mp4'), fps=FPS)

        return True
    except Exception as e:
        print(f"Error processing {sub_name}: {str(e)}")
        return False

def main(start_from: int = 0):
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # populate subdirectories
    subject = [
        name for name in os.listdir(PURE_PATH)
        if os.path.isdir(os.path.join(PURE_PATH, name)) and re.search(r'\d$', name)
    ]
    subject.sort()
    
    # Start from a specific subject
    subject = subject[start_from:]
    print(f"Subjects: {subject}")
    print(f"Total subjects: {len(subject)}")

    facial_extractor = FacialExtractor()

    successful = []
    failed = []

    for sub_name in subject:
        print(f"Processing {sub_name} | {subject.index(sub_name)+1}/{len(subject)}")
        if process_subject(sub_name, facial_extractor):
            successful.append(sub_name)
        else:
            failed.append(sub_name)

    # Generate report
    with open(REPORT_FILE, 'w') as f:
        f.write("Successfully processed videos:\n")
        for sub in successful:
            f.write(f"- {sub}\n")
        f.write("\nFailed to process videos:\n")
        for sub in failed:
            f.write(f"- {sub}\n")

    print(f"Processing complete. Report saved to {REPORT_FILE}")

if __name__ == '__main__':
    main(start_from=18)