import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix, roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from thop import profile, clever_format
from tqdm import tqdm
import os

# === CONFIGURATIONS ===
MODEL_PATH = './logs/dual_rafdb/dual_RAF-DB_best.pt'
CSV_SAVE_DIR = './logs/dual_rafdb/'
BATCH_SIZE = 64
NUM_CLASSES = 7
CLASS_NAMES = ['Happy', 'Sad', 'Angry', 'Neutral', 'Surprise', 'Fear', 'Disgust']  # Edit sesuai dataset Anda
INPUT_SIZE = (224, 224)

# === 1. Load Model ===
model = torch.load(MODEL_PATH, map_location='cpu',weights_only=True)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# === 2. Load Test Dataset ===
test_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
])
test_dataset = datasets.ImageFolder('path_to_test_folder', transform=test_transform)  # Ubah path test folder
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === 3. Hitung Params dan FLOPs ===
dummy_input = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1]).to(device)
flops, params = profile(model, inputs=(dummy_input,), verbose=False)
flops, params = clever_format([flops, params], '%.3f')

# === 4. Evaluasi Model ===
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Evaluasi"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)

        all_preds.append(outputs.argmax(dim=1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_probs = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)

# === 5. Hitung Metrics ===
acc = accuracy_score(all_labels, all_preds)
kappa = cohen_kappa_score(all_labels, all_preds)

top1_correct = (all_preds == all_labels).sum()
top1_acc = top1_correct / len(all_labels)

top3_preds = np.argsort(all_probs, axis=1)[:, -3:]
top3_correct = np.any(top3_preds == all_labels.reshape(-1,1), axis=1).sum()
top3_acc = top3_correct / len(all_labels)

report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
cm = confusion_matrix(all_labels, all_preds)
classwise_acc = cm.diagonal() / cm.sum(axis=1)
uac = np.mean(classwise_acc)

# === 6. Save CSV Results ===
os.makedirs(CSV_SAVE_DIR, exist_ok=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(CSV_SAVE_DIR, 'classification_report.csv'))

summary = {
    'Accuracy': acc,
    'Top-1 Accuracy': top1_acc,
    'Top-3 Accuracy': top3_acc,
    'Kappa': kappa,
    'UAC': uac,
    'Params': params,
    'FLOPs': flops
}
summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(CSV_SAVE_DIR, 'evaluation_summary.csv'), index=False)

# === 7. Save Confusion Matrix ===
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(CSV_SAVE_DIR, 'confusion_matrix.png'))
plt.close()

# === 8. ROC Curve per Class ===
plt.figure(figsize=(10,8))
for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve per Class')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CSV_SAVE_DIR, 'roc_curve.png'))
plt.close()

print("✅ Evaluasi selesai. Semua hasil disimpan di:", CSV_SAVE_DIR)
