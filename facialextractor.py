import torch
import torchvision
import dlib
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

class FacialExtractor:
    def __init__(self):
        # Initiate Sam Model
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam = sam.to(device=device)
        self.predictor = SamPredictor(sam)
        
    def detect_face_bbox(self, image):
        # Load the detector
        detector = dlib.get_frontal_face_detector()

        # Ensure the image is in the correct format
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            if len(image.shape) == 2:
                # It's already grayscale
                gray = image
            elif len(image.shape) == 3:
                # It's a color image, convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        elif isinstance(image, str):
            # If it's a file path
            image = cv2.imread(image)
            if image is None:
                raise ValueError("Could not read the image file")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Ensure the image is contiguous in memory
        if not gray.flags['C_CONTIGUOUS']:
            gray = np.ascontiguousarray(gray)

        # Print debug information
        print(f"Gray image shape: {gray.shape} | Dtype: {gray.dtype}")

        try:
            # Detect faces
            faces = detector(gray, 1)
        except RuntimeError as e:
            print(f"Error during face detection: {str(e)}")
            print("Attempting to use OpenCV's face detector as a fallback...")
            
            # Fallback to OpenCV's face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                return (x, y, x+w, y+h)
            else:
                return None

        # Check if any faces were detected
        if len(faces) > 0:
            # Get the first face (assuming single face detection)
            face = faces[0]
            
            # Get face coordinates
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            
            return (x1, y1, x2, y2)
        else:
            return None