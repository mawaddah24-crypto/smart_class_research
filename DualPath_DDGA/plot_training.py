import pandas as pd
import matplotlib.pyplot as plt

# Load log hasil training
log = pd.read_csv("./logs/baseline_rafdb/baseline_RAF-DB_log.csv")

# Setup plot
plt.figure(figsize=(12, 5))

# Plot akurasi
plt.subplot(1, 2, 1)
plt.plot(log['epoch'], log['train_loss'], label='Train Loss', marker='o')
plt.plot(log['epoch'], log['val_loss'], label='Val Loss', marker='s')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

# Tampilkan plot
plt.tight_layout()
plt.show()
