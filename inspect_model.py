import os
import joblib
import pandas as pd

MODEL_PATH = './output/drowsiness_model.pkl'

def inspect():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train_model.py first!")

    print(f"--- Loading model from {MODEL_PATH} ---")
    model = joblib.load(MODEL_PATH)
    print("Model successfully loaded!")

    print("\n--- Model Specifications ---")
    print(f"Model Type: {type(model).__name__}")
    print(f"Number of Trees: {getattr(model, 'n_estimators', 'N/A')}")
    print(f"Expected Features: {getattr(model, 'n_features_in_', 'N/A')}")
    print(f"Classes: {getattr(model, 'classes_', 'N/A')} (0: Alert, 1: Drowsy)")

    feature_cols = ['ear', 'mar', 'pitch', 'yaw', 'roll']

    if hasattr(model, 'feature_importances_'):
        print("\n--- Feature Importances ---")
        for name, importance in zip(feature_cols, model.feature_importances_):
            print(f" - {name.upper():6s}: {importance * 100:.2f}%")

    print("\n--- Testing Model Inferences ---")

    # Sample 1: Alert (Eyes Open, Mouth Closed, Normal Head Pose)
    alert_df = pd.DataFrame([[0.30, 0.12, 0.0, 0.0, 0.0]], columns=feature_cols)
    prob_alert = model.predict_proba(alert_df)[0][1]
    pred_alert = model.predict(alert_df)[0]

    # Sample 2: Drowsy (Eyes Closed, Mouth Normal)
    drowsy_df = pd.DataFrame([[0.12, 0.15, 5.0, 0.0, 0.0]], columns=feature_cols)
    prob_drowsy = model.predict_proba(drowsy_df)[0][1]
    pred_drowsy = model.predict(drowsy_df)[0]

    print(f"Alert Sample  -> Drowsiness Prob: {prob_alert * 100:.2f}% | Prediction: {pred_alert}")
    print(f"Drowsy Sample -> Drowsiness Prob: {prob_drowsy * 100:.2f}% | Prediction: {pred_drowsy}")

    if pred_drowsy == 1 and pred_alert == 0:
        print("\nAll checks passed! The model reliably differentiates Alert vs Drowsy.")
    else:
        print("\nWarning: Model predictions seem biased. Ensure you trained with class_weight='balanced'.")

if __name__ == '__main__':
    inspect()