import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import argparse

# Daftar label FER 7 kelas standar
LABELS = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']

def analyze(csv_path):
    df = pd.read_csv(csv_path)
    print(f"📄 Loaded {len(df)} samples from {csv_path}")

    y_true = df['label'].astype(str)

    # Prediksi = argmax dari skor probabilitas
    y_pred = df[LABELS].idxmax(axis=1)

    # Akurasi total
    acc = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Overall Accuracy: {acc:.4f}")

    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=LABELS))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=LABELS, yticklabels=LABELS, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Akurasi per kelas
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=LABELS, y=per_class_acc)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help="Path to classification CSV file")
    args = parser.parse_args()

    analyze(args.csv)

if __name__ == "__main__":
    main()
