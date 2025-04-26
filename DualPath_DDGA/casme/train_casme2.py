import os
import time
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import model dari DualPathModel.py
from DualPathModel import (
    DualPath_Baseline,
    DualPath_Baseline_DSE,
    DualPath_PartialAttentionModif,
    DualPath_PartialAttentionSAP
)

def get_data_loaders(data_dir, batch_size):
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

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

def build_model(model_name, num_classes, backbone_pretrained=True):
    if model_name == "baseline":
        model = DualPath_Baseline(num_classes=num_classes, pretrained=backbone_pretrained)
    elif model_name == "dse":
        model = DualPath_Baseline_DSE(num_classes=num_classes, pretrained=backbone_pretrained)
    elif model_name == "partialmodif":
        model = DualPath_PartialAttentionModif(num_classes=num_classes, pretrained=backbone_pretrained)
    elif model_name == "partialsap":
        model = DualPath_PartialAttentionSAP(num_classes=num_classes, pretrained=backbone_pretrained)
    else:
        raise ValueError(f"Model {model_name} tidak dikenali!")
    return model

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size)
    model = build_model(args.model, args.num_classes, backbone_pretrained=True)
    model = model.to(device)

    # Load backbone pretrained weight (EfficientViT) jika ada
    if args.backbone_weights and os.path.exists(args.backbone_weights):
        print(f"🔹 Load backbone pretrained: {args.backbone_weights}")
        state_dict = torch.load(args.backbone_weights, map_location=device)
        model.load_backbone_weights(state_dict)

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
            torch.save(best_model_wts, os.path.join(args.output_dir, f"best_{args.model}_casme2.pth"))
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
    parser = argparse.ArgumentParser(description="Training CASME2 Apex with DualPath Model")
    parser.add_argument('--data_dir', type=str, required=True, help="Path ke dataset CASME2 Apex")
    parser.add_argument('--output_dir', type=str, required=True, help="Path ke output log/model")
    parser.add_argument('--backbone_weights', type=str, default=None, help="Path ke pretrained backbone EfficientViT")
    parser.add_argument('--model', type=str, required=True, choices=["baseline", "dse", "partialmodif", "partialsap"], help="Pilih model")
    parser.add_argument('--num_classes', type=int, default=5, help="Jumlah kelas CASME2 Apex")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--epochs', type=int, default=100, help="Jumlah epoch")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--early_stop', type=int, default=10, help="Patience untuk early stopping")
    return parser.parse_args()

def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()
