import os
import sys
import logging
import warnings
import torch
import cv2
import numpy as np
import dlib
import mediapipe as mp
from segment_anything import sam_model_registry, SamPredictor
from mtcnn import MTCNN

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
    def __init__(self, device='cuda'):
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
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

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

    def crop_img(self, image, bbox, expandbbox=1.6):
        """
        Crop and expand the face image based on the detected bounding box.

        Args:
            image (numpy.ndarray): Input image in RGB format.
            bbox (tuple): Bounding box coordinates (left, top, right, bottom).
            expandbbox (float): Factor to expand the bounding box. Default is 1.6.

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
        
        # Expand the bounding box size by 1.2 times
        size = int(size * expandbbox)
        
        # Ensure the expanded box stays within the image bounds
        new_x1 = max(0, center_x - size // 2)
        new_y1 = max(0, center_y - size // 2)
        new_x2 = min(image.shape[1], new_x1 + size)
        new_y2 = min(image.shape[0], new_y1 + size)
        
        # Crop The Image
        face_image = image[new_y1:new_y2, new_x1:new_x2]
        
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
        interest_point = [(int(forehead_point.x * width), int(forehead_point.y * height)),
                          (int(cheek_point_r.x * width), int(cheek_point_r.y * height)),
                          (int(cheek_point_l.x * width), int(cheek_point_l.y * height))]
        
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

        # Initialize SAM model
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        predictor = SamPredictor(sam)
        
        input_point = np.array(interest_point)
        # Set the input label based on length of input point
        input_label = np.ones(len(input_point))
        predictor.set_image(image)
        
        masks, score, logits = predictor.predict(
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