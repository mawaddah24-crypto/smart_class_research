import os
import pandas as pd
import matplotlib.pyplot as plt

# Folder lokasi semua logs
log_dir = "./logs/"

# Model dan dataset yang mau dibaca
experiments = [
    ("baseline_rafdb", "DualPath_Baseline", "RAF-DB"),
    ("baseline_fer2013", "DualPath_Baseline", "FER2013"),
    ("partial_rafdb", "DualPath_PartialAttention", "RAF-DB"),
    ("partial_fer2013", "DualPath_PartialAttention", "FER2013"),
    ("partialmodif_rafdb", "DualPath_PartialAttentionModif", "RAF-DB"),
    ("partialmodif_fer2013", "DualPath_PartialAttentionModif", "FER2013")
]

# Buat summary kosong
summary_data = []

# Membaca semua log file
for exp_name, model_name, dataset_name in experiments:
    log_path = os.path.join(log_dir, exp_name, f"{exp_name}_log.csv")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        best_row = df.loc[df['val_acc'].idxmax()]
        summary_data.append({
            "Model": model_name,
            "Dataset": dataset_name,
            "Best Val Acc (%)": round(best_row['val_acc'], 2),
            "Best Epoch": int(best_row['epoch']),
            "Train Loss": round(best_row['train_loss'], 4),
            "Val Loss": round(best_row['val_loss'], 4)
        })
    else:
        print(f"⚠️ Log file not found: {log_path}")

# Buat summary table
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("./logs/summary_result.csv", index=False)
print("✅ Summary saved to ./logs/summary_result.csv")

# Tampilkan summary
print(summary_df)
# Plot Training Curve untuk masing-masing model
plt.figure(figsize=(12, 8))

for exp_name, model_name, dataset_name in experiments:
    log_path = os.path.join(log_dir, exp_name, f"{exp_name}_log.csv")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        plt.plot(df['epoch'], df['val_acc'], label=f"{model_name} ({dataset_name})")

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Validation Accuracy Curve for DualPath Models")
plt.legend()
plt.grid(True)
plt.savefig("./logs/val_accuracy_curves.png")
print("✅ Validation Accuracy Curve saved to ./logs/val_accuracy_curves.png")
plt.show()
