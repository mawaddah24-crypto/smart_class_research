import os, csv
import torch
import argparse
import torch.amp
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import pandas as pd
import numpy as np
from timm.scheduler import CosineLRScheduler
from DualPathAdaptive import DualPathAdaptivePlus
from loaders import load_pretrained_backbone
from FocalLoss import FocalLoss

# === Cosine Warmup Scheduler ===
class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535)))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]

# Augmentasi MixUp & CutMix
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size).to(x.device)
    rx = np.random.randint(W)
    ry = np.random.randint(H)
    rw = int(W * np.sqrt(1 - lam))
    rh = int(H * np.sqrt(1 - lam))
    x[:, :, ry:ry+rh, rx:rx+rw] = x[index, :, ry:ry+rh, rx:rx+rw]
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

# Backbone Freezer
def set_backbone_trainable(model, trainable: bool):
    for param in model.backbone.parameters():
        param.requires_grad = trainable
    print(f"{'🔥 Unfrozen' if trainable else '🧊 Frozen'} backbone")

# Training Function
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model = DualPathAdaptivePlus(num_classes=args.num_classes, pretrained=True).to(device)
    
    
    checkpoint_path = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_last.pt')
    base_path = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_best.pt')
    log_file = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_log.csv')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=10, max_epochs=args.epochs, min_lr=1e-5)

    if args.backbone_weights:
        load_pretrained_backbone(model, args.backbone_weights)
        
    criterion = FocalLoss(gamma=2.0)
    scaler = GradScaler(device='cuda')

    # Data Augmentation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, args.dataset, 'train'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, args.dataset, 'test'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    start_epoch = 0
    best_acc = 0
    history = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        if epoch < 3:
            set_backbone_trainable(model, False)
        else:
            set_backbone_trainable(model, True)

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]", unit='batch')
        for batch_idx, (imgs, labels) in enumerate(loop):
            imgs, labels = imgs.to(device), labels.to(device)

            if np.random.rand() < 0.5:
                if np.random.rand() < 0.5:
                    imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, alpha=0.2)
                else:
                    imgs, labels_a, labels_b, lam = cutmix_data(imgs, labels, alpha=0.3)
            else:
                labels_a = labels_b = labels
                lam = 1.0

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(imgs)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

        train_acc = 100. * correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]", unit="batch"):
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)
        scheduler.step()

        print(f"\nEpoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss_avg:.4f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_acc': best_acc
        }, checkpoint_path)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), base_path)
            print(f"✅ Model Terbaik di Simpan Acc: {best_acc:.2f}%")

        # Log
        row = {"epoch": epoch+1, "train_loss": running_loss / len(train_loader),
               "val_loss": val_loss_avg, "val_loss_acc": val_loss, "val_acc": val_acc}
        history.append(row)
        pd.DataFrame(history).to_csv(log_file, index=False)

# Entry Point
if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='RAF-DB')
    parser.add_argument('--data_dir', type=str, default='../dataset_cropped_yolo/')
    parser.add_argument("--output_dir", type=str, default="adaptive")
    parser.add_argument("--backbone_weights", type=str, default="./weights/efficientvit_vggface2_best.pth")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument('--model', type=str, default='semfusion', choices=['semfusion', 'baseline'])
    args = parser.parse_args()
    print(f"✅ Konfigurasi Model: {args}")
    train(args)
