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
    def __init__(self, device='cuda'):
        self.device = device
        
        # Set up face detector
        self.detector = dlib.get_frontal_face_detector()
        
        # Set up MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    def detect_face_bbox(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray, 1)
        if len(faces) > 0:
            face = faces[0]
            return (face.left(), face.top(), face.right(), face.bottom())
        return None

    def crop_img(self, image, bbox, expandbbox=1.6):
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