import os
import glob
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
DATA_DIR = './Data'
OUTPUT_CSV = './data/drowsiness_features.csv'
MODEL_PATH = 'face_landmarker.task'

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ==========================================
# 2. METRIC CALCULATION HELPERS
# ==========================================
def euclidean_dist(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def calculate_ear(landmarks, eye_indices):
    v1 = euclidean_dist(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    v2 = euclidean_dist(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    h = euclidean_dist(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)

def calculate_mar(landmarks, mouth_indices):
    v = euclidean_dist(landmarks[mouth_indices[1]], landmarks[mouth_indices[2]])
    h = euclidean_dist(landmarks[mouth_indices[0]], landmarks[mouth_indices[3]])
    if h == 0:
        return 0.0
    return v / h

def estimate_head_pose(landmarks, img_w, img_h):
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye corner
        (225.0, 170.0, -135.0),      # Right eye corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)

    idx = [1, 152, 33, 263, 61, 291]
    image_points = np.array([
        (landmarks[i][0] * img_w, landmarks[i][1] * img_h) for i in idx
    ], dtype=np.float64)

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))
    
    success, rotation_vec, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    return angles[0], angles[1], angles[2]

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 13, 14, 308]

# ==========================================
# 3. INITIALIZE MEDIAPIPE LANDMARKER
# ==========================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model task file '{MODEL_PATH}' not found in current directory.")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# ==========================================
# 4. GATHER DATASET IMAGES & LABELS
# ==========================================
print(f"Scanning '{DATA_DIR}' for images...")

image_paths = (
    glob.glob(os.path.join(DATA_DIR, '**', '*.png'), recursive=True) +
    glob.glob(os.path.join(DATA_DIR, '**', '*.jpg'), recursive=True) +
    glob.glob(os.path.join(DATA_DIR, '**', '*.jpeg'), recursive=True)
)

print(f"Found {len(image_paths)} total images.")

features_data = []

for idx, img_path in enumerate(image_paths):
    # Relative path normalized to lowercase
    rel_path = os.path.relpath(img_path, DATA_DIR).lower()
    
    # Matching Logic: Check Non Drowsy first!
    if any(k in rel_path for k in ['non drowsy', 'nondrowsy', 'non_drowsy', 'alert', 'open', 'normal', 'active']):
        label = 0  # Alert / Non-Drowsy
    elif any(k in rel_path for k in ['drowsy', 'yawn', 'closed', 'sleep']):
        label = 1  # Drowsy
    else:
        continue  # Skip unclassified images

    try:
        image = cv2.imread(img_path)
        if image is None:
            continue

        img_h, img_w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks]

            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            
            mar = calculate_mar(landmarks, MOUTH)
            pitch, yaw, roll = estimate_head_pose(landmarks, img_w, img_h)

            features_data.append({
                'ear': avg_ear,
                'mar': mar,
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll,
                'label': label
            })
    except Exception:
        continue

    if (idx + 1) % 1000 == 0 or (idx + 1) == len(image_paths):
        print(f"Processed {idx + 1}/{len(image_paths)} images...")

# ==========================================
# 5. SAVE DATASET TO CSV
# ==========================================
df = pd.DataFrame(features_data)

if not df.empty:
    df.to_csv(OUTPUT_CSV, index=False)
    print("\n------------------------------------------------")
    print(f"SUCCESS! Dataset saved to '{OUTPUT_CSV}'")
    print(f"Total labeled samples extracted: {len(df)}")
    print(f"Class Distribution:\n{df['label'].value_counts()}")
    print("------------------------------------------------")
else:
    print("\n[ERROR] No valid face features extracted.")