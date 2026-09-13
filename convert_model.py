import joblib
import numpy as np
from skl2onnx import to_onnx

# 1. Load your model using joblib
model = joblib.load(r'output\drowsiness_model.pkl')

# 2. Define dummy input matching 5 features
dummy_input = np.zeros((1, 5), dtype=np.float32)

# 3. Convert to ONNX format
onnx_model = to_onnx(model, dummy_input)

# 4. Save file in output folder
output_path = r'output\drowsiness_model.onnx'
with open(output_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Successfully generated {output_path}!")