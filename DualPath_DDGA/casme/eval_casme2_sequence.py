import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

from casme2_sequence_dataset import CASME2SequenceDataset
from dualpath_lstm import DualPathLSTM

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_dataset = CASME2SequenceDataset(root_dir=os.path.join(args.data_dir, "val"), seq_len=args.seq_len, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = DualPathLSTM(num_classes=args.num_classes, feature_dim=args.feature_dim, lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    print("\n🎯 Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=val_dataset.classes, digits=4))

    print("\n🧩 Confusion Matrix:\n")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CASME2 Sequence Model")
    parser.add_argument('--data_dir', type=str, required=True, help="Path ke dataset sequences split")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path model checkpoint (.pth)")
    parser.add_argument('--num_classes', type=int, default=5, help="Jumlah kelas")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size untuk evaluasi")
    parser.add_argument('--seq_len', type=int, default=18, help="Panjang sequence")
    parser.add_argument('--feature_dim', type=int, default=256, help="Dimensi fitur backbone")
    parser.add_argument('--lstm_hidden', type=int, default=128, help="Hidden dimensi LSTM")
    parser.add_argument('--lstm_layers', type=int, default=1, help="Jumlah layer LSTM")
    args = parser.parse_args()

    evaluate(args)
