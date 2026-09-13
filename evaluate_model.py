import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the actual dataset from the Data folder
csv_path = r'Data\drowsiness_features.csv'

try:
    data = pd.read_csv(csv_path)
    print(f"Dataset successfully loaded from '{csv_path}'!")
except FileNotFoundError:
    print(f"Error: Could not find '{csv_path}'. Check the path!")
    exit()

# 2. Inspect target column name dynamically (e.g., 'label', 'target', 'state', or 'drowsy')
# Drops target column for X, extracts target column for y
possible_target_cols = ['label', 'target', 'state', 'drowsy', 'is_drowsy', 'Class', 'class']
target_col = None

for col in possible_target_cols:
    if col in data.columns:
        target_col = col
        break

if target_col is None:
    # If no standard name matches, assume the last column is the label
    target_col = data.columns[-1]

print(f"Using target column: '{target_col}'")

X = data.drop(columns=[target_col])
y = data[target_col]

# 3. Perform test split (using 20% test data to match training validation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Load trained model
model_path = r'output\drowsiness_model.pkl'
model = joblib.load(model_path)

# 5. Predict & display accuracy metrics
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 45)
print(f" Model Test Accuracy: {acc * 100:.2f}%")
print("=" * 45)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))