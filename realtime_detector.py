import math
import os
import sys
import threading
import time
from collections import deque
from typing import Any

import cv2  # type: ignore
import joblib
import numpy as np
import pandas as pd
import pyttsx3
import mediapipe as mp  # type: ignore

# MediaPipe 0.10.35 Face Mesh Module Binding
try:
    mp_face_mesh = mp.solutions.face_mesh  # type: ignore
except AttributeError as err:
    print("\n[ERROR] Failed to bind MediaPipe Face Mesh.")
    print(f"Details: {err}\n")
    sys.exit(1)

# 1. Load Trained ML Model
POSSIBLE_PATHS = [
    './output/drowsiness_model.pkl',
    'drowsiness_model.pkl',
    'output/drowsiness_model.pkl'
]

loaded_model: Any = None
for model_path in POSSIBLE_PATHS:
    if os.path.exists(model_path):
        try:
            loaded_model = joblib.load(model_path)
            print(f"Machine Learning Model Loaded Successfully from {model_path}.")
            break
        except Exception as load_err:
            print(f"Error loading model from {model_path}: {load_err}")

if loaded_model is None:
    print("Error: Could not locate or load 'drowsiness_model.pkl'.")
    sys.exit(1)

model: Any = loaded_model


# 2. Non-Blocking Voice Alert System
class VoiceAlertManager:
    def __init__(self):
        self.speech_queue = deque()
        self.is_speaking = False
        self.lock = threading.Lock()

        # Start background worker thread
        self.thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.thread.start()

    def _speech_worker(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
        except Exception as init_err:
            print(f"Voice alert initialization warning: {init_err}")
            return

        while True:
            text = None
            with self.lock:
                if self.speech_queue:
                    text = self.speech_queue.popleft()
                    self.is_speaking = True

            if text:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as speak_err:
                    print(f"Speech synthesis error: {speak_err}")
                with self.lock:
                    self.is_speaking = False
            else:
                time.sleep(0.05)

    def speak(self, text):
        with self.lock:
            if not self.is_speaking and len(self.speech_queue) == 0:
                self.speech_queue.append(text)


voice_system = VoiceAlertManager()

# 3. MediaPipe Landmark Indices & Metric Functions
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 0, 17, 13, 14]

FEATURE_COLS = ['ear', 'mar', 'pitch', 'yaw', 'roll']


def calculate_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def calculate_ear(landmarks, eye_indices):
    p = [landmarks[i] for i in eye_indices]
    vertical_dist1 = calculate_distance(p[1], p[5])
    vertical_dist2 = calculate_distance(p[2], p[4])
    horizontal_dist = calculate_distance(p[0], p[3])
    return (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist + 1e-6)


def calculate_mar(landmarks, mouth_indices):
    p = [landmarks[i] for i in mouth_indices]
    vertical_inner = calculate_distance(p[2], p[3])
    vertical_outer = calculate_distance(p[4], p[5])
    horizontal_width = calculate_distance(p[0], p[1])
    return (vertical_inner + vertical_outer) / (2.0 * horizontal_width + 1e-6)


def estimate_head_pose(landmarks, image_shape):
    h, w, _ = image_shape
    face_3d = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),  # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ], dtype=np.float64)

    face_2d = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),
        (landmarks[152].x * w, landmarks[152].y * h),
        (landmarks[263].x * w, landmarks[263].y * h),
        (landmarks[33].x * w, landmarks[33].y * h),
        (landmarks[291].x * w, landmarks[291].y * h),
        (landmarks[61].x * w, landmarks[61].y * h)
    ], dtype=np.float64)

    focal_length = 1.0 * w
    cam_matrix = np.array([[focal_length, 0, w / 2],
                           [0, focal_length, h / 2],
                           [0, 0, 1]], dtype=np.float64)
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    pitch = angles[0] * 360
    yaw = angles[1] * 360
    roll = angles[2] * 360
    return pitch, yaw, roll


# 4. Main Real-time Processing Loop
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    ear_buffer = deque(maxlen=10)
    mar_buffer = deque(maxlen=10)

    drowsy_frame_count = 0
    yawn_frame_count = 0
    last_alert_time = 0.0

    drowsy_threshold_frames = 15
    yawn_threshold_frames = 20

    print("\n--- Realtime Driver Monitoring System Online ---")
    print("Press 'q' to exit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        status_text = "Status: Alert"
        status_color = (0, 255, 0)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(landmarks, MOUTH)
            pitch, yaw, roll = estimate_head_pose(landmarks, (h, w, c))

            ear_buffer.append(ear)
            mar_buffer.append(mar)
            avg_ear = sum(ear_buffer) / len(ear_buffer)
            avg_mar = sum(mar_buffer) / len(mar_buffer)

            sample_df = pd.DataFrame([[avg_ear, avg_mar, pitch, yaw, roll]], columns=FEATURE_COLS)

            model_prob = model.predict_proba(sample_df)[0][1]
            model_pred = model.predict(sample_df)[0]

            is_eyes_closed = avg_ear < 0.18 or model_pred == 1
            is_yawning = avg_mar > 0.55

            if is_eyes_closed:
                drowsy_frame_count += 1
            else:
                drowsy_frame_count = max(0, drowsy_frame_count - 1)

            if is_yawning:
                yawn_frame_count += 1
            else:
                yawn_frame_count = max(0, yawn_frame_count - 1)

            current_time = time.time()

            if drowsy_frame_count >= drowsy_threshold_frames:
                status_text = "ALERT: DROWSINESS DETECTED!"
                status_color = (0, 0, 255)
                if current_time - last_alert_time > 3.0:
                    voice_system.speak("Warning! Please wake up!")
                    last_alert_time = current_time

            elif yawn_frame_count >= yawn_threshold_frames:
                status_text = "WARNING: FREQUENT YAWNING"
                status_color = (0, 255, 255)
                if current_time - last_alert_time > 4.0:
                    voice_system.speak("You are yawning. Consider taking a break.")
                    last_alert_time = current_time

            cv2.putText(frame, f"EAR: {avg_ear:.2f} | MAR: {avg_mar:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Drowsy Prob: {model_prob * 100:.1f}%", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        else:
            cv2.putText(frame, "No Face Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Driver Drowsiness Monitoring System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()