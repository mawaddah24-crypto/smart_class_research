import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (classification_report, confusion_matrix, cohen_kappa_score,
                             accuracy_score, top_k_accuracy_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from thop import profile, clever_format

from FocalLoss import FocalLoss
from DualPath_SemFusion import DualPath_SemFusion
from DualPathModel import DualPath_Baseline, DualPath_Fusion

def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    preds, targets, losses, all_probs = [], [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc='Evaluating', leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            probs = F.softmax(outputs, dim=1)
            preds.append(outputs)
            all_probs.append(probs)
            targets.append(labels)
            losses.append(loss.item())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    probs = torch.cat(all_probs)
    preds_top1 = torch.argmax(probs, dim=1)
    preds_top3 = torch.topk(probs, k=3, dim=1).indices

    return preds_top1.cpu(), preds_top3.cpu(), targets.cpu(), probs.cpu(), np.mean(losses)

def save_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_prob, n_classes, save_path):
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr, tpr, roc_auc = dict(), dict(), dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.plot(fpr["macro"], tpr["macro"], color='navy', lw=3, linestyle='--',
             label=f'Macro-Average (AUC = {roc_auc["macro"]:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Multi-Class)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return roc_auc

def main():
    parser = argparse.ArgumentParser(description='Evaluate FER Model')
    parser.add_argument('--model', type=str,  default='baseline', choices=['semfusion', 'baseline', 'fusion','base_ddga'],
                        help='Choose model: semfusion | baseline | fusion')
    
    parser.add_argument('--dataset', type=str, default='RAF-DB', help='Dataset folder name')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = (224, 224)
    batch_size = 64
    n_classes = 7
    test_path = os.path.join('../dataset_cropped_yolo/', args.dataset, 'test')
    if len(test_path) == 0:
        raise FileNotFoundError(f"❌ Path dataset tidak ditemukan: {test_path}")
    
   
    # Load Model
    if args.model == 'semfusion':
        model = DualPath_SemFusion(num_classes=n_classes)
        checkpoint_path = './logs/v1/base_rafdb/baseline_RAF-DB_last.pt'
    elif args.model == 'baseline':
        model = DualPath_Baseline(num_classes=n_classes)
        checkpoint_path = './logs/v1/base_rafdb/baseline_RAF-DB_best.pt'
    elif args.model == 'base_ddga':
        model = DualPath_Baseline(num_classes=n_classes)
        model.delay_ddga=False
        checkpoint_path = './logs/v1/base_ddga_rafdb/baseline_RAF-DB_best.pt'
    elif args.model == 'fusion':
        model = DualPath_Fusion(num_classes=n_classes)
        checkpoint_path = './logs/v1/fusion_rafdb/fusion_RAF-DB_best.pt'
    else:
        raise ValueError("Model tidak dikenali.")

    state = torch.load(checkpoint_path, map_location=device,weights_only=True)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    model.to(device)

    # Dataset
    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])
    test_dataset = datasets.ImageFolder(test_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"📊 Val samples: {len(test_dataset)}")
    
    # FLOPs & Params
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], '%.3f')

    # Loss
    criterion = FocalLoss(gamma=2.0)
    save_dir = 'logs/evaluation_logs/gradcam'
    os.makedirs(save_dir, exist_ok=True)
    # Evaluation
    top1, top3, targets, probs, loss = evaluate_model(model, test_loader, device, criterion)

    acc1 = accuracy_score(targets, top1)
    #acc3 = top_k_accuracy_score(targets, top3, k=3)
    acc3 = top_k_accuracy_score(targets, probs, k=3)
    kappa = cohen_kappa_score(targets, top1)
    report = classification_report(targets, top1, digits=4)
    cm = confusion_matrix(targets, top1)
    roc_auc = plot_roc_curve(targets.numpy(), probs.numpy(), n_classes,
                              save_path=f'{save_dir}/roc_curve_{args.model}.png')

    # Save results
    
    save_confusion_matrix(cm, [str(i) for i in range(n_classes)], f'{save_dir}/confusion_matrix_{args.model}.png')

    with open(f'{save_dir}/eval_{args.model}.txt', 'w') as f:
        f.write(f"Validation Loss: {loss:.4f}\n")
        f.write(f"Accuracy Top-1: {acc1:.4f}\n")
        f.write(f"Accuracy Top-3: {acc3:.4f}\n")
        f.write(f"Cohen Kappa: {kappa:.4f}\n")
        f.write(f"FLOPs: {flops}\nParams: {params}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nAUROC per class:\n")
        for i in range(n_classes):
            f.write(f"Class {i}: AUC = {roc_auc[i]:.4f}\n")
        f.write(f"Macro-Average AUROC = {roc_auc['macro']:.4f}\n")

    print("✅ Evaluation complete. Results saved in 'evaluation_logs/'.")

if __name__ == '__main__':
    main()
