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
    ):
        """
        Initialize the FacialExtractor with specified device.

        Args:
            device (str): The device to use for computations. Default is 'cuda'.
        """
        self.device = device

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

    def crop_img(self, image, bbox, expandbbox, out_size):
        """
        Crop and expand the face image based on the detected bounding box.

        Args:
            image (numpy.ndarray): Input image in RGB format.
            bbox (tuple): Bounding box coordinates (left, top, right, bottom).
            expandbbox (float): Factor to expand the bounding box.
            out_size (int): Size of the output face image.

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
        size = int(size * expandbbox)

        # Ensure the expanded box stays within the image bounds
        new_x1 = max(0, center_x - size // 2)
        new_y1 = max(0, center_y - size // 2)
        new_x2 = min(image.shape[1], new_x1 + size)
        new_y2 = min(image.shape[0], new_y1 + size)

        # Crop The Image
        face_image = image[new_y1:new_y2, new_x1:new_x2]

        # resize the image
        face_image = cv2.resize(face_image, (out_size, out_size))
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

    def track_img_frames(
        self,
        images,
        use_klt=True,
        dlib_interval=1,
        winSize=(15, 15),
        maxLevel=2,
        minDistance=5,
        maxCorners=150,
        smoothing_window=0,
    ):
        """
        Track the face movement in a series of image frames using KLT tracker or dlib.

        Args:
            images (numpy.ndarray): 4D numpy array of images (num_frames x height x width x channels)
            use_klt (bool): Whether to use KLT tracking. If False, use dlib for face detection.
            dlib_interval (int): Interval for face detection when not using KLT.
            winSize (tuple): Size of the search window at each pyramid level for KLT.
            maxLevel (int): 0-based maximal pyramid level number for KLT.
            minDistance (int): Minimum possible Euclidean distance between the returned corners for goodFeaturesToTrack.
            maxCorners (int): Maximum number of corners to return for goodFeaturesToTrack.
            smoothing_window (int): Size of the moving average window for smoothing. If 0, no smoothing is applied.

        Returns:
            list: List of bounding boxes for each frame.
        """
        if not isinstance(images, np.ndarray) or images.ndim != 4:
            raise ValueError("Input must be a 4D numpy array of images.")

        bboxes = []
        old_gray = None
        p0 = None
        lk_params = dict(
            winSize=winSize,
            maxLevel=maxLevel,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        initial_bbox = None
        bbox_size = None

        for i in range(images.shape[0]):
            image_rgb = images[i]
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

            if i == 0 or (not use_klt and i % dlib_interval == 0):
                # Use dlib to detect the face
                bbox = self.detect_face_bbox(image_rgb)
                if bbox is None:
                    if i == 0:
                        raise ValueError("No face detected in the first frame.")
                    else:
                        # If face detection fails, use the last known bbox
                        bbox = bboxes[-1]

                initial_bbox = bbox
                x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                bbox_size = (w, h)

                if use_klt:
                    mask = np.zeros_like(image_gray)
                    mask[y : y + h, x : x + w] = 255
                    p0 = cv2.goodFeaturesToTrack(
                        image_gray,
                        mask=mask,
                        maxCorners=maxCorners,
                        qualityLevel=0.01,
                        minDistance=minDistance,
                    )

                old_gray = image_gray.copy()
            elif use_klt:
                if (
                    p0 is None or len(p0) < 4
                ):  # We need at least 4 points for a bounding box
                    # If we lost too many points, re-detect the face
                    bbox = self.detect_face_bbox(image_rgb)
                    if bbox is None:
                        # If face detection fails, use the last known bbox
                        bbox = bboxes[-1]
                    x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                    mask = np.zeros_like(image_gray)
                    mask[y : y + h, x : x + w] = 255
                    p0 = cv2.goodFeaturesToTrack(
                        image_gray,
                        mask=mask,
                        maxCorners=maxCorners,
                        qualityLevel=0.01,
                        minDistance=minDistance,
                    )
                else:
                    # Calculate optical flow
                    p1, st, err = cv2.calcOpticalFlowPyrLK(
                        old_gray, image_gray, p0, None, **lk_params
                    )

                    # Select good points
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]

                    # Calculate new bounding box
                    if len(good_new) >= 4:
                        # Calculate the movement of the center of the tracked points
                        old_center = np.mean(good_old, axis=0)
                        new_center = np.mean(good_new, axis=0)
                        movement = new_center - old_center

                        # Update the bounding box position based on the movement
                        x, y = (
                            initial_bbox[0] + movement[0],
                            initial_bbox[1] + movement[1],
                        )
                        bbox = (
                            int(x),
                            int(y),
                            int(x + bbox_size[0]),
                            int(y + bbox_size[1]),
                        )
                    else:
                        # If we have too few points, use the last known bbox
                        bbox = bboxes[-1]

                    # Update points for next iteration
                    p0 = good_new.reshape(-1, 1, 2)

                old_gray = image_gray.copy()
            else:
                # If not using KLT and not on a dlib interval frame, use the last known bbox
                bbox = bboxes[-1]

            # Apply smoothing if window size is greater than 0
            if smoothing_window > 0:
                start = max(0, i - smoothing_window + 1)
                window = bboxes[start:] + [bbox]
                bbox = tuple(map(lambda x: int(sum(x) / len(x)), zip(*window)))

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

        fig, axs = plt.subplots(
            num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows)
        )
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
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")

            # Add the patch to the Axes
            axs[i].add_patch(rect)
            axs[i].set_title(f"Frame {idx}")
            axs[i].axis("off")

        # Remove any unused subplots
        for i in range(num_frames, len(axs)):
            fig.delaxes(axs[i])

        plt.tight_layout()
        plt.show()

    def process_img_frames(
        self,
        image_path,
        use_klt=True,
        dlib_interval=1,
        landmark_interval=0.5,
        segment_interval=1.0,
        fps=30,
        expandbbox=2.0,
        out_size=224,
        verbose=False,
        winSize=(15, 15),
        maxLevel=2,
        minDistance=5,
        maxCorners=150,
        smoothing_window=0,
    ):
        """
        Process a series of image frames to extract face skin.

        Args:
            image_path (str): Path to the directory containing image frames.
            use_klt (bool): Whether to use KLT tracking. If False, use dlib for face detection.
            dlib_interval (int): Interval for face detection when not using KLT.
            landmark_interval (float): Interval in seconds to detect facial landmarks.
            segment_interval (float): Interval in seconds to perform face segmentation.
            fps (int): Frames per second of the input video.
            expandbbox (float): Factor to expand the bounding box.
            out_size (int): Size of the output face image.
            verbose (bool): Whether to print progress information.
            winSize (tuple): Size of the search window at each pyramid level for KLT.
            maxLevel (int): 0-based maximal pyramid level number for KLT.
            minDistance (int): Minimum possible Euclidean distance between the returned corners for goodFeaturesToTrack.
            maxCorners (int): Maximum number of corners to return for goodFeaturesToTrack.
            smoothing_window (int): Size of the moving average window for smoothing bounding boxes. If 0, no smoothing is applied.

        Returns:
            numpy.ndarray: 4D array of extracted face skin images (num_frames x height x width x channels)
        """
        # 0. Read the images
        if verbose:
            print(f"Reading images from {image_path}...")
            time_start = time.time()
        img_files = sorted(glob(os.path.join(image_path, "*.*")))
        img_files = [f for f in img_files if f.endswith(("jpg", "jpeg", "png"))]

        if verbose:
            print(f"Found {len(img_files)} image files.")
            print(f"Time taken: {time.time() - time_start:.2f} seconds")

        images = []
        for img_file in img_files:
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

        # convert the list into numpy array
        images = np.array(images)

        # 1. Obtain the bounding boxes for each frame
        if verbose:
            print("Tracking face movement in the image frames...")
            time_start = time.time()
        bboxes = self.track_img_frames(
            images,
            use_klt=use_klt,
            dlib_interval=dlib_interval,
            winSize=winSize,
            maxLevel=maxLevel,
            minDistance=minDistance,
            maxCorners=maxCorners,
            smoothing_window=smoothing_window,
        )

        if verbose:
            print(f"Time taken: {time.time() - time_start:.2f} seconds")

        extracted_faces = []
        last_landmark_frame = -1
        last_segment_frame = -1
        last_landmarks = None
        last_mask = None

        landmark_frame_interval = int(landmark_interval * fps)
        segment_frame_interval = int(segment_interval * fps)

        for i, (image, bbox) in enumerate(zip(images, bboxes)):
            # display progress

            # 2. Crop the face from each frame using `crop_img`
            time_start = time.time()
            face_image = self.crop_img(image, bbox, expandbbox, out_size)
            time_crop = time.time() - time_start

            # 3. Detect facial landmarks every `landmark_interval` seconds
            if (
                i - last_landmark_frame >= landmark_frame_interval
                or last_landmarks is None
            ):
                time_start = time.time()
                landmarks = self.detect_lm(face_image)
                last_landmark_frame = i
                last_landmarks = landmarks
                time_landmark = time.time() - time_start
            else:
                landmarks = last_landmarks
                time_landmark = 0

            if landmarks is None:
                print(
                    f"Warning: No landmarks detected for frame {i}. Skipping this frame."
                )
                continue

            # 4. Segment the face every `segment_interval` seconds
            if i - last_segment_frame >= segment_frame_interval or last_mask is None:
                time_start = time.time()
                masks, _, _ = self.segment_face(face_image, landmarks)
                last_segment_frame = i
                last_mask = masks
                time_segment = time.time() - time_start
            else:
                masks = last_mask
                time_segment = 0

            # 5. Extract the face skin from each frame using `extract_face_skin`
            time_start = time.time()
            face_skin = self.extract_face_skin(face_image, masks)

            extracted_faces.append(face_skin)
            time_extract = time.time() - time_start

            # Verbose
            if i % 173 == 0 and verbose:
                # in milisecond
                print(
                    f"Frame {i} / {len(images)}| Crop: {time_crop*1000:.2f} ms | Landmark: {time_landmark*1000:.2f} ms | Segment: {time_segment*1000:.2f} ms | Extract: {time_extract*1000:.2f} ms"
                )

        # 6. Return the extracted face skin images
        return np.array(extracted_faces)
