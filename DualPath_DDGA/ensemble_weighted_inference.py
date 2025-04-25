import os
import torch
import torch.nn.functional as F
from DualPathModel import DualPath_PartialAttentionModif
import re

# === SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_class = DualPath_PartialAttentionModif  # Sesuaikan dengan model Anda
checkpoint_dir = "./logs/partialmodif_rafdb/"
num_classes = 7
top_k = 5  # Bisa diubah menjadi 3, 5, dst

# === BACA SEMUA CHECKPOINT DENGAN AKURASI ===
checkpoints = []
for fname in os.listdir(checkpoint_dir):
    if fname.endswith(".pt") and "_acc" in fname:
        match = re.search(r"acc(\d+\.\d+)", fname)
        if match:
            acc = float(match.group(1))
            path = os.path.join(checkpoint_dir, fname)
            checkpoints.append((acc, path))

# Ambil top-k
checkpoints = sorted(checkpoints, key=lambda x: x[0], reverse=True)[:top_k]

# === LOAD MODEL CHECKPOINT ===
models = []
weights = []
for acc, path in checkpoints:
    model = model_class(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    models.append(model)
    weights.append(acc)

# Normalisasi bobot berdasarkan akurasi
total = sum(weights)
weights = [w / total for w in weights]

# === DUMMY INPUT ===
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# === ENSEMBLE FORWARD ===
outputs = []
for model, w in zip(models, weights):
    with torch.no_grad():
        logits = model(dummy_input)
        outputs.append(logits * w)

avg_logits = torch.stack(outputs).sum(dim=0)
predicted = torch.argmax(F.softmax(avg_logits, dim=1), dim=1)

print(f"✅ Predicted Class: {predicted.item()}")
