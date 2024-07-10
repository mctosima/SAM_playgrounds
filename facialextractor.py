import os
import time
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from glob import glob
import sys
import logging
import warnings
import torch
import cv2
import numpy as np
import dlib
import mediapipe as mp
from segment_anything import sam_model_registry, SamPredictor

# surpress warnings
warnings.filterwarnings("ignore")


class FacialExtractor:
    """
    A class for facial feature extraction, detection, and segmentation.

    This class provides methods for detecting facial bounding boxes, cropping face images,
    detecting facial landmarks, and segmenting face skin using the Segment Anything Model (SAM).

    Attributes:
        device (str): The device to use for computations ('cuda' or 'cpu').
        detector: A dlib face detector object.
        mp_face_mesh: MediaPipe FaceMesh solution.
        face_mesh: MediaPipe FaceMesh model for facial landmark detection.

    Methods:
        detect_face_bbox(image): Detect face bounding box in an image.
        crop_img(image, bbox, expandbbox): Crop and expand face image based on bounding box.
        detect_lm(image): Detect facial landmarks using MediaPipe.
        segment_face(image, interest_point): Segment face using SAM.
        extract_face_skin(image, mask, resize): Extract face skin based on segmentation mask.
    """

    def __init__(
        self,
        device="cuda",
        expandbbox=1.6,
        out_size=224,
    ):
        """
        Initialize the FacialExtractor with specified device.

        Args:
            device (str): The device to use for computations. Default is 'cuda'.
        """
        self.device = device
        self.expandbbox = expandbbox
        self.out_size = out_size

        # Set up face detector
        self.detector = dlib.get_frontal_face_detector()

        # Set up MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
        )
        
        # Initialize SAM model
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    def detect_face_bbox(self, image):
        """
        Detect face bounding box in the given image using dlib.

        Args:
            image (numpy.ndarray): Input image in RGB format.

        Returns:
            tuple: Bounding box coordinates (left, top, right, bottom) or None if no face detected.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray, 1)
        if len(faces) > 0:
            face = faces[0]
            return (face.left(), face.top(), face.right(), face.bottom())
        return None

    def crop_img(self, image, bbox):
        """
        Crop and expand the face image based on the detected bounding box.

        Args:
            image (numpy.ndarray): Input image in RGB format.
            bbox (tuple): Bounding box coordinates (left, top, right, bottom).

        Returns:
            numpy.ndarray: Cropped and expanded face image.
        """

        # Ensure the image is in RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3 and image.dtype == np.uint8:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Expand the bounding box so that it will be square (according to the longer side)
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        size = max(width, height)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # manual adjustment for center_y
        center_y = int(center_y - 0.2 * size)

        # Expand the bounding box size
        size = int(size * self.expandbbox)

        # Ensure the expanded box stays within the image bounds
        new_x1 = max(0, center_x - size // 2)
        new_y1 = max(0, center_y - size // 2)
        new_x2 = min(image.shape[1], new_x1 + size)
        new_y2 = min(image.shape[0], new_y1 + size)

        # Crop The Image
        face_image = image[new_y1:new_y2, new_x1:new_x2]
        
        # resize the image
        face_image = cv2.resize(face_image, (self.out_size, self.out_size))
        return face_image

    def detect_lm(self, image):
        """
        Detect facial landmarks using MediaPipe FaceMesh.

        Args:
            image (numpy.ndarray): Input face image in RGB format.

        Returns:
            list: List of tuples containing coordinates of interest points (forehead, right cheek, left cheek),
                  or None if no face landmarks detected.
        """

        results = self.face_mesh.process(image)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        forehead_point = landmarks[4]
        cheek_point_r = landmarks[425]
        cheek_point_l = landmarks[50]

        height, width, _ = image.shape
        interest_point = [
            (int(forehead_point.x * width), int(forehead_point.y * height)),
            (int(cheek_point_r.x * width), int(cheek_point_r.y * height)),
            (int(cheek_point_l.x * width), int(cheek_point_l.y * height)),
        ]

        return interest_point

    def segment_face(self, image, interest_point):
        """
        Segment the face using the Segment Anything Model (SAM).

        Args:
            image (numpy.ndarray): Input face image in RGB format.
            interest_point (list): List of tuples containing coordinates of interest points.

        Returns:
            tuple: Contains three elements:
                - masks (numpy.ndarray): Binary mask of the segmented face.
                - score (float): Confidence score of the segmentation.
                - logits (numpy.ndarray): Raw logits from the SAM model.
        """



        input_point = np.array(interest_point)
        # Set the input label based on length of input point
        input_label = np.ones(len(input_point))
        self.predictor.set_image(image)

        masks, score, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        return masks, score, logits

    def extract_face_skin(self, image, mask, resize=224):
        """
        Extract the face skin based on the segmentation mask.

        Args:
            image (numpy.ndarray): Input face image in RGB format.
            mask (numpy.ndarray): Binary mask of the segmented face.
            resize (int): Size to resize the output image. Default is 224. Set to None for no resizing.

        Returns:
            numpy.ndarray: Extracted face skin image, optionally resized.
        """

        # if the mask is 3 dimension, unsqueeze the first dimension
        if len(mask.shape) == 3:
            mask = mask[0]

        # Convert the mask to binary
        mask = mask > 0.5

        # Extract the face skin
        face_skin = np.zeros_like(image)
        face_skin[mask] = image[mask]

        if resize is not None:
            face_skin = cv2.resize(face_skin, (resize, resize))

        return face_skin

    def track_img_frames(self, path):
        """
        Track the face movement in a series of image frames using KLT tracker.

        Args:
            path (str): Path to the directory containing image frames.

        Returns:
            list: List of bounding boxes for each frame.
        """
        if not os.path.isdir(path):
            raise ValueError("The path is not a directory.")

        img_files = sorted(glob(os.path.join(path, "*.*")))
        img_files = [f for f in img_files if f.endswith(("jpg", "jpeg", "png"))]
        if len(img_files) == 0:
            raise ValueError("No image files found in the directory.")

        bboxes = []
        old_gray = None
        p0 = None
        lk_params = dict(winSize=(15,15), maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        initial_bbox = None
        bbox_size = None

        for i, img_path in enumerate(img_files):
            image_bgr = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

            if i == 0:
                # For the first frame, use dlib to detect the face
                bbox = self.detect_face_bbox(image_rgb)
                if bbox is None:
                    raise ValueError("No face detected in the first frame.")
                
                initial_bbox = bbox
                x, y, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
                bbox_size = (w, h)
                
                mask = np.zeros_like(image_gray)
                mask[y:y+h, x:x+w] = 255

                # Set tracking points
                p0 = cv2.goodFeaturesToTrack(image_gray, mask=mask, maxCorners=100, 
                                            qualityLevel=0.3, minDistance=7, blockSize=7)
                
                old_gray = image_gray.copy()
                bboxes.append(bbox)
            else:
                # For subsequent frames, use KLT tracking
                if p0 is None or len(p0) < 4:  # We need at least 4 points for a bounding box
                    # If we lost too many points, re-detect the face
                    bbox = self.detect_face_bbox(image_rgb)
                    if bbox is None:
                        # If face detection fails, use the last known bbox
                        bbox = bboxes[-1]
                    x, y, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
                    mask = np.zeros_like(image_gray)
                    mask[y:y+h, x:x+w] = 255
                    p0 = cv2.goodFeaturesToTrack(image_gray, mask=mask, maxCorners=100, 
                                                qualityLevel=0.3, minDistance=7, blockSize=7)
                else:
                    # Calculate optical flow
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, image_gray, p0, None, **lk_params)

                    # Select good points
                    good_new = p1[st==1]
                    good_old = p0[st==1]

                    # Calculate new bounding box
                    if len(good_new) >= 4:
                        # Calculate the movement of the center of the tracked points
                        old_center = np.mean(good_old, axis=0)
                        new_center = np.mean(good_new, axis=0)
                        movement = new_center - old_center

                        # Update the bounding box position based on the movement
                        x, y = initial_bbox[0] + movement[0], initial_bbox[1] + movement[1]
                        bbox = (int(x), int(y), int(x + bbox_size[0]), int(y + bbox_size[1]))
                    else:
                        # If we have too few points, use the last known bbox
                        bbox = bboxes[-1]

                    # Update points for next iteration
                    p0 = good_new.reshape(-1, 1, 2)

                old_gray = image_gray.copy()
                bboxes.append(bbox)

        return bboxes
    
    def preview_tracking(self, image_paths, bboxes, num_frames=20):
        """
        Preview the tracking results using matplotlib, distributing frames in a grid layout.

        Args:
            image_paths (list): List of paths to image frames.
            bboxes (list): List of bounding boxes corresponding to each frame.
            num_frames (int): Number of frames to preview. Default is 20.

        Returns:
            None
        """
        num_frames = min(num_frames, len(image_paths))
        
        # Calculate the number of rows and columns
        num_cols = min(5, num_frames)  # Max 5 columns
        num_rows = math.ceil(num_frames / num_cols)
        
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
        if num_rows == 1 and num_cols == 1:
            axs = np.array([axs])
        axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easy indexing

        for i in range(num_frames):
            # Select frames evenly spread across the sequence
            idx = i * len(image_paths) // num_frames
            img = plt.imread(image_paths[idx])
            axs[i].imshow(img)
            
            # Get the bounding box
            bbox = bboxes[idx]
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            # Create a Rectangle patch
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            
            # Add the patch to the Axes
            axs[i].add_patch(rect)
            axs[i].set_title(f'Frame {idx}')
            axs[i].axis('off')

        # Remove any unused subplots
        for i in range(num_frames, len(axs)):
            fig.delaxes(axs[i])

        plt.tight_layout()
        plt.show()
        
    def process_img_frames(self, path):
        """
        Process a series of image frames to extract face skin.

        Args:
            path (str): Path to the directory containing image frames.

        Returns:
            numpy.ndarray: 4D array of extracted face skin images (num_frames x height x width x channels)
        """
        # 1. Obtain the bounding boxes for each frame
        bboxes = self.track_img_frames(path)

        # Get list of image files
        img_files = sorted(glob(os.path.join(path, "*.*")))
        img_files = [f for f in img_files if f.endswith(("jpg", "jpeg", "png"))]

        extracted_faces = []

        for i, (img_path, bbox) in enumerate(zip(img_files, bboxes)):
            # display progress
            if i % 20 == 0:
                print(f"Processing frame {i} from {len(img_files)} frames...")
            
            # Read the image
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 2. Crop the face from each frame using `crop_img`
            face_image = self.crop_img(image, bbox)

            # 3. Detect facial landmarks for every frame using `detect_lm`
            landmarks = self.detect_lm(face_image)

            if landmarks is None:
                print(f"Warning: No landmarks detected for frame {i}. Skipping this frame.")
                continue

            # 4. Segment every frame the face using `segment_face`
            masks, _, _ = self.segment_face(face_image, landmarks)

            # 5. Extract the face skin from each frame using `extract_face_skin`
            face_skin = self.extract_face_skin(face_image, masks)

            extracted_faces.append(face_skin)

        # 6. Return the extracted face skin images
        return np.array(extracted_faces)