import cv2
import numpy as np
import os
import urllib.request

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

class HandDetector:
    def __init__(self):
        self.use_tasks = False
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
                self.mp_hands = mp.solutions.hands
                self.mp_drawing = mp.solutions.drawing_utils
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )
            else:
                self.use_tasks = True
        except Exception:
            self.use_tasks = True

        if self.use_tasks:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            model_path = "hand_landmarker.task"
            if not os.path.exists(model_path):
                print("Downloading hand_landmarker.task model...")
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                urllib.request.urlretrieve(url, model_path)
                
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.mp = mp

    def process(self, frame_bgr):
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        if not self.use_tasks:
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                return results.multi_hand_landmarks[0].landmark
            return None
        else:
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
            res = self.detector.detect(mp_image)
            if res.hand_landmarks and len(res.hand_landmarks) > 0:
                return res.hand_landmarks[0]
            return None

    def draw_landmarks(self, frame, landmarks):
        h, w, _ = frame.shape
        coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        
        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(frame, coords[p1], coords[p2], (0, 255, 0), 2)
            
        for x, y in coords:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
