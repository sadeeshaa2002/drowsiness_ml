import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = './data/drowsiness_features.csv'
OUTPUT_DIR = './output'
OUTPUT_MODEL_PATH = os.path.join(OUTPUT_DIR, 'drowsiness_model.pkl')

def train():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset missing at {DATA_PATH}. Please verify the folder structure.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("--- Loading Dataset ---")
    df = pd.read_csv(DATA_PATH)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()
    
    feature_cols = ['ear', 'mar', 'pitch', 'yaw', 'roll']
    X = df[feature_cols]
    y = df['label']

    print(f"Dataset loaded: {len(df):,} samples.")
    print("Class Distribution:\n", y.value_counts(normalize=True))

    # Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n--- Training Balanced Random Forest Classifier ---")
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        class_weight='balanced',  # Forces balanced weights across Alert & Drowsy classes
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Performance Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n================ EVALUATION RESULTS ================")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Alert', 'Drowsy']))

    # Feature Importances Breakdown
    print("Feature Importance Breakdown:")
    for name, importance in zip(feature_cols, model.feature_importances_):
        print(f"  {name.upper():6s}: {importance * 100:.2f}%")

    # Save Artifact
    joblib.dump(model, OUTPUT_MODEL_PATH)
    print(f"\nSUCCESS! Optimized model saved to '{OUTPUT_MODEL_PATH}'")

if __name__ == '__main__':
    train()