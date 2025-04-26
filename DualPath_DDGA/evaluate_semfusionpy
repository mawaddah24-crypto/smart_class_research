import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, top_k_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

from dataset import YourTestDataset  # Ganti dengan dataset Anda
from DualPath_SemFusion import DualPath_SemFusion  # Import model Anda

def evaluate(model, dataloader, device, criterion):
    model.eval()
    preds = []
    targets = []
    losses = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating', leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            preds.append(outputs)
            targets.append(labels)
            losses.append(loss.item())

            pbar.set_postfix({"Batch Loss": loss.item()})

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    probs = F.softmax(preds, dim=1)
    preds_top1 = torch.argmax(probs, dim=1)
    preds_top3 = torch.topk(probs, k=3, dim=1).indices

    avg_loss = np.mean(losses)

    return preds_top1, preds_top3, targets, avg_loss

def save_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Load Model ===
    model = DualPath_SemFusion(num_classes=7)
    model.load_state_dict(torch.load('./checkpoint_best.pth', map_location=device))  # Sesuaikan path checkpoint
    model.to(device)

    # === Load Test Dataset ===
    test_dataset = YourTestDataset(...)  # Isi dengan test dataset Anda
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # === Loss function ===
    criterion = nn.CrossEntropyLoss()

    # === Evaluation ===
    preds_top1, preds_top3, targets, val_loss = evaluate(model, test_loader, device, criterion)

    preds_top1 = preds_top1.cpu().numpy()
    preds_top3 = preds_top3.cpu().numpy()
    targets = targets.cpu().numpy()

    # === Metrics ===
    acc_top1 = accuracy_score(targets, preds_top1)
    acc_top3 = top_k_accuracy_score(targets, preds_top3, k=3)
    kappa = cohen_kappa_score(targets, preds_top1)
    report = classification_report(targets, preds_top1, digits=4)
    cm = confusion_matrix(targets, preds_top1)

    # === Print Metrics (Standard Report Style) ===
    print(f"\nValidation Loss: {val_loss:.4f}")
    print(f"Accuracy: {acc_top1:.4f}")
    print(f"Accuracy Top-1: {acc_top1:.4f}")
    print(f"Accuracy Top-3: {acc_top3:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print("\nClassification Report:\n", report)

    # === Save Metrics ===
    os.makedirs('evaluation_logs', exist_ok=True)

    with open('evaluation_logs/eval_report.txt', 'w') as f:
        f.write(f"Validation Loss: {val_loss:.4f}\n")
        f.write(f"Accuracy: {acc_top1:.4f}\n")
        f.write(f"Accuracy Top-1: {acc_top1:.4f}\n")
        f.write(f"Accuracy Top-3: {acc_top3:.4f}\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)

    save_confusion_matrix(cm, class_names=[str(i) for i in range(7)], save_path='evaluation_logs/confusion_matrix.png')
    print("✅ Evaluation results saved in 'evaluation_logs/'.")

if __name__ == '__main__':
    main()
