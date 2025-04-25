import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from DualPathModel import DualPath_Baseline
from DualPathUnifiedInferenceModel import DualPathUnifiedInferenceModel
from WeightedPartialAttention import WeightedPartialAttention
from dummy_gaze_pose_generator import generate_dummy_gaze_pose_importance

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model  = DualPath_Baseline(num_classes=7)
base_model.load_state_dict(torch.load("../logs/baseline_rafdb/baseline_RAF-DB_best.pt", map_location=device,weights_only=True))
base_model.to(device)
base_model.eval()
# Build Unified Inference Model
unified_model = DualPathUnifiedInferenceModel(base_model )
unified_model.to(device)
unified_model.eval()

# Dummy input
input_tensor = torch.randn(1, 3, 224, 224).to(device)

# Inference
with torch.no_grad():
    prediction = unified_model(input_tensor)
    pred_class = prediction.argmax(dim=1)
    print(f"Predicted Expression Class: {pred_class.item()}")