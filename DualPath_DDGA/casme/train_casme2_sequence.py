import os
import time
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from casme2_sequence_dataset import CASME2SequenceDataset
from dualpath_lstm import DualPathLSTM

def get_data_loaders(data_dir, batch_size, seq_len):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CASME2SequenceDataset(root_dir=os.path.join(data_dir, "train"), seq_len=seq_len, transform=train_transforms)
    val_dataset = CASME2SequenceDataset(root_dir=os.path.join(data_dir, "val"), seq_len=seq_len, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size, args.seq_len)
    model = DualPathLSTM(num_classes=args.num_classes, feature_dim=args.feature_dim, lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_acc = 0.0
    patience_counter = 0

    print("\n🔵 Starting Training...\n")
    for epoch in range(args.epochs):
        start_time = time.time()

        # Train
        model.train()
        running_loss, running_corrects = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, val_corrects = 0.0, 0

        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        scheduler.step(val_loss)

        # Logging
        print(f"🧪 Epoch {epoch+1}/{args.epochs} -- Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} || Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} || Time: {time.time()-start_time:.2f}s")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(args.output_dir, f"best_dualpath_lstm_casme2.pth"))
            print(f"✅ Best model updated! Val Acc: {best_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⏳ EarlyStopping patience: {patience_counter}/{args.early_stop}")

        if patience_counter >= args.early_stop:
            print("⛔ Early stopping triggered.")
            break

    print("\n🎉 Training Completed.")

def parse_args():
    parser = argparse.ArgumentParser(description="Training CASME2 Sequences with DualPath + LSTM")
    parser.add_argument('--data_dir', type=str, required=True, help="Path dataset sequences CASME2")
    parser.add_argument('--output_dir', type=str, required=True, help="Path output logs/models")
    parser.add_argument('--num_classes', type=int, default=5, help="Jumlah kelas ekspresi")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--epochs', type=int, default=100, help="Jumlah epoch")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--early_stop', type=int, default=10, help="Patience EarlyStopping")
    parser.add_argument('--seq_len', type=int, default=18, help="Panjang sequence frame")
    parser.add_argument('--feature_dim', type=int, default=256, help="Dimensi fitur dari DualPath")
    parser.add_argument('--lstm_hidden', type=int, default=128, help="Hidden dimensi LSTM")
    parser.add_argument('--lstm_layers', type=int, default=1, help="Jumlah layer LSTM")
    return parser.parse_args()

def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()
