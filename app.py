import cv2
import numpy as np
import joblib
import os
import time
from hand_tracker import HandDetector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.joblib")

# Loading the trained model
print(f"Loading trained model from '{MODEL_PATH}'...")
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure you have run 'python train_model.py' first.")
    exit(1)

detector = HandDetector()

def extract_features(hand_landmarks):
    coords = []
    for lm in hand_landmarks:
        coords.append([lm.x, lm.y, lm.z])
        
    coords = np.array(coords)
    
    # Translation-center at wrist
    wrist = coords[0].copy()
    translated = coords - wrist
    
    # Scaling-normalize by max distance from wrist
    max_dist = np.max(np.linalg.norm(translated, axis=1))
    if max_dist > 0:
        normalized = translated / max_dist
    else:
        normalized = translated
        
    return normalized.flatten().reshape(1, -1)

# Start Video Capture
print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Index 0 failed, trying index 1...")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Warm up camera - discard initial dark frames
print("Warming up camera...")
for i in range(30):
    cap.read()
    time.sleep(0.03)

ret, test_frame = cap.read()
if ret:
    h, w = test_frame.shape[:2]
    print(f"Camera ready: {w}x{h}")
else:
    print("Warning: Could not read test frame, continuing anyway...")

print("Starting video translation. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        cv2.waitKey(30)
        continue

    frame = cv2.flip(frame, 1)
    landmarks = detector.process(frame)

    if landmarks:
        detector.draw_landmarks(frame, landmarks)
        
        # Predict
        features = extract_features(landmarks)
        prediction = model.predict(features)[0]
        confidence_probs = model.predict_proba(features)[0]
        confidence = max(confidence_probs)
        
        h, w, c = frame.shape
        x_min = int(min([lm.x for lm in landmarks]) * w) - 20
        y_min = int(min([lm.y for lm in landmarks]) * h) - 20
        
        # Keep text within bounds
        x_min = max(0, x_min)
        y_min = max(30, y_min)
        
        color = (0, 255, 0) if confidence >= 0.5 else (0, 0, 255)
        text = f"{prediction} ({confidence*100:.0f}%)"
        
        cv2.putText(frame, text, (x_min, y_min), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    # Instructions text
    cv2.putText(frame, "SignSpeak AI - Press 'q' to Exit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('SignSpeak AI - Real-time Detection', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
