import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, top_k_accuracy_score
from thop import profile
from pathlib import Path

# Import model
from DualPathModel import (
    DualPath_Baseline,
    DualPath,
    DualPath_Baseline_DSE,
    DualPath_PartialAttentionModif,
    DualPath_PartialAttentionSAP,
    DualPath_Base_Partial  # standard
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name, num_classes):
    if model_name == "baseline":
        model = DualPath_Baseline(num_classes=num_classes, pretrained=True)
    elif model_name == "baseline_dse":
        model = DualPath_Baseline_DSE(num_classes=num_classes, pretrained=True)
    elif model_name == "partial":
        model = DualPath_Base_Partial(num_classes=num_classes, pretrained=True)
    elif model_name == "partialmodif":
        model = DualPath_PartialAttentionModif(num_classes=num_classes, pretrained=True)
    elif model_name == "partialsap":
        model = DualPath_PartialAttentionSAP(num_classes=num_classes, pretrained=True)
    elif model_name == "dual":
        model = DualPath(num_classes=num_classes, pretrained=True)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

def evaluate(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc_top1 = accuracy_score(all_labels, all_preds)
    acc_top3 = top_k_accuracy_score(all_labels, torch.softmax(outputs, dim=1).cpu().numpy(), k=3)
    kappa = cohen_kappa_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    return acc_top1, acc_top3, kappa, report, cm

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_flops_params(model, input_size=(3, 224, 224)):
    dummy_input = torch.randn(1, *input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    return flops, params

def main():
    model_names = ['dual',"baseline", "baseline_dse", "partial", "partialmodif", "partialsap"]
    datasets_info = {
        "rafdb": {
            "num_classes": 7,
            "path": "../dataset_cropped_yolo/rafdb/test"
        }
    }

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for dataset_name, dataset_info in datasets_info.items():
        print(f"\n📚 Evaluating on {dataset_name.upper()}")

        data_dir = dataset_info["path"]
        num_classes = dataset_info["num_classes"]

        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        class_names = dataset.classes

        for model_name in tqdm(model_names, desc="Evaluating Models"):
            print(f"\n🚀 Evaluating Model: {model_name}")

            output_folder = f"./eval_results/{dataset_name}/{model_name}/"
            os.makedirs(output_folder, exist_ok=True)

            model = load_model(model_name, num_classes=num_classes)
            checkpoint_path = f"./logs/{model_name}_{dataset_name}/{model_name.upper()}_{dataset_name.upper()}_best.pt"

            if not os.path.exists(checkpoint_path):
                print(f"⚠️ Skipped {model_name} on {dataset_name}: checkpoint not found at {checkpoint_path}")
                continue

            print(f"✅ Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model = model.to(device)

            # Evaluation
            acc_top1, acc_top3, kappa, report, cm = evaluate(model, dataloader, class_names)
            flops, params = calculate_flops_params(model)

            # Save confusion matrix
            plot_confusion_matrix(cm, class_names, save_path=os.path.join(output_folder, "confusion_matrix.png"))

            # Save report
            with open(os.path.join(output_folder, "report.txt"), "w") as f:
                f.write(f"Dataset: {dataset_name.upper()}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Top-1 Accuracy: {acc_top1:.4f}\n")
                f.write(f"Top-3 Accuracy: {acc_top3:.4f}\n")
                f.write(f"Cohen's Kappa: {kappa:.4f}\n")
                f.write(f"Total Parameters: {params/1e6:.2f} M\n")
                f.write(f"FLOPs: {flops/1e9:.2f} GFLOPs\n\n")
                f.write("Classification Report:\n")
                f.write(report)
                f.write("\nConfusion Matrix:\n")
                f.write(np.array2string(cm))

if __name__ == "__main__":
    main()
