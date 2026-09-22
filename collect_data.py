import cv2
import mediapipe as mp
import numpy as np
import os
import csv

LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
          'OK', 'Stop', 'thankyou', 'help', 'i love you', 'hello', 'sorry', 'yes', 'no']

# Number of frames to capture per class 
SAMPLES_PER_CLASS = 150

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def collect_data():
    output_path = "dataset.csv"
    
    # We will overwrite the old dataset
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)
        
        cap = cv2.VideoCapture(0)
        
        for label in LABELS:
            print(f"Get ready for sign: {label}")
            capturing = False
            frames_captured = 0
            
            while True:
                ret, frame = cap.read()
                if not ret: continue
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                # Draw hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        if capturing:
                            # Extract raw coords without normalization 
                            row = [label]
                            for lm in hand_landmarks.landmark:
                                # We scale it up (pseudo reverse-normalization) just to keep numbers large 
                                
                                row.extend([lm.x * 500, lm.y * 500, lm.z * 500])
                            writer.writerow(row)
                            frames_captured += 13

                # Display Instructions
                if not capturing:
                    cv2.putText(frame, f"Sign: '{label}'", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(frame, "Get ready! Press 's' to start capturing.", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, "Press 'skip' (k) to skip this sign.", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, f"Capturing '{label}': {frames_captured}/{SAMPLES_PER_CLASS}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("SignSpeak AI - Data Collector", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s') and not capturing:
                    capturing = True
                elif key == ord('k') and not capturing:
                    print(f"Skipping {label}")
                    break
                elif key == ord('q'):
                    print("Exiting...")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                
                if frames_captured >= SAMPLES_PER_CLASS:
                    print(f"Finished capturing {label}!")
                    # Pause briefly
                    cv2.putText(frame, "Done!", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    cv2.imshow("SignSpeak AI - Data Collector", frame)
                    cv2.waitKey(1000)
                    break
                    
        cap.release()
        cv2.destroyAllWindows()
        print(f"Real WebCam dataset collected and saved to {output_path}!")
        print("Now you can run 'python train_model.py' to train on your real hand data!")

if __name__ == "__main__":
    collect_data()
