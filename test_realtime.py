import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load Trained Model
MODEL_PATH = './output/drowsiness_model.pkl'
LANDMARKER_PATH = 'face_landmarker.task'

model = joblib.load(MODEL_PATH)

# Setup MediaPipe
base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Helper Functions
def euclidean_dist(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def calculate_ear(landmarks, eye_indices):
    v1 = euclidean_dist(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    v2 = euclidean_dist(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    h = euclidean_dist(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    return (v1 + v2) / (2.0 * h) if h != 0 else 0.0

def calculate_mar(landmarks, mouth_indices):
    v = euclidean_dist(landmarks[mouth_indices[1]], landmarks[mouth_indices[2]])
    h = euclidean_dist(landmarks[mouth_indices[0]], landmarks[mouth_indices[3]])
    return v / h if h != 0 else 0.0

def estimate_head_pose(landmarks, img_w, img_h):
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ], dtype=np.float64)

    idx = [1, 152, 33, 263, 61, 291]
    image_points = np.array([(landmarks[i][0] * img_w, landmarks[i][1] * img_h) for i in idx], dtype=np.float64)

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1], angles[2]

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 13, 14, 308]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_h, img_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    if result.face_landmarks:
        landmarks = [(lm.x, lm.y, lm.z) for lm in result.face_landmarks[0]]

        ear = (calculate_ear(landmarks, LEFT_EYE) + calculate_ear(landmarks, RIGHT_EYE)) / 2.0
        mar = calculate_mar(landmarks, MOUTH)
        pitch, yaw, roll = estimate_head_pose(landmarks, img_w, img_h)

        features = np.array([[ear, mar, pitch, yaw, roll]])
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][prediction]

        label = "DROWSY!" if prediction == 1 else "ALERT"
        color = (0, 0, 255) if prediction == 1 else (0, 255, 0)

        cv2.putText(frame, f"Status: {label} ({prob*100:.1f}%)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f} | MAR: {mar:.2f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Driver Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()