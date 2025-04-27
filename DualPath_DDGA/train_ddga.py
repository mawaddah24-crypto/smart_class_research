# train_hcavit_fastmulti.py

import os, csv
import torch
import argparse
import torch.amp
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from tqdm import tqdm
import pandas as pd
import numpy as np
from DualPathModel import DualPath_DDGA, DualPath_Fusion, DualPath_DRM
from loaders import load_pretrained_backbone
#from FERLandmarkDataset import FERLandmarkCachedDataset  # Sesuaikan ini
from FocalLoss import FocalLoss
from AdaptiveEntropyController import AdaptiveEntropyController

# Inisialisasi controller
entropy_controller = AdaptiveEntropyController(
    start_lambda=0.01, 
    min_lambda=0.002, 
    decay_epochs=(15, 30, 50)
)
# --------------------------
# Fungsi Augmentasi MixUp & CutMix
# --------------------------
def mixup_data(x, y, alpha=1.0):
    """MixUp Augmentasi"""
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
    """CutMix Augmentasi"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size).to(x.device)

    # Tentukan lokasi cutout
    rx = np.random.randint(W)
    ry = np.random.randint(H)
    rw = int(W * np.sqrt(1 - lam))
    rh = int(H * np.sqrt(1 - lam))

    x[:, :, ry:ry+rh, rx:rx+rw] = x[index, :, ry:ry+rh, rx:rx+rw]
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam

def set_backbone_trainable(model, trainable: bool):
    for param in model.backbone.parameters():
        param.requires_grad = trainable
    print(f"{'🔥 Unfrozen' if trainable else '🧊 Frozen'} backbone")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    # 🔧 Inisialisasi model
    if args.model == "fusion":
        model = DualPath_Fusion(num_classes=args.num_classes, pretrained=True)
    elif args.model == "drm":
        model = DualPath_DRM(num_classes=args.num_classes, pretrained=True)
    else:
        model = DualPath_DDGA(num_classes=args.num_classes, pretrained=True)
        
    model.to(device)
    
    checkpoint_path = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_last.pt')
    base_path = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_best.pt')
    log_file = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_log.csv')
    
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # 🧠 Load EfficientViT pretrained dari VGGFace2
    if args.backbone_weights:
        load_pretrained_backbone(model, args.backbone_weights)

    #criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(gamma=2.0)
    scaler = GradScaler(device='cuda')

    # 📦 Dataset
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Konversi ke Tensor harus di awal!
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_path = os.path.join(args.data_dir, args.dataset, 'train')
    val_path = os.path.join(args.data_dir, args.dataset, 'test')
    if len(train_path) == 0 or len(val_path) == 0:
        raise FileNotFoundError(f"❌ Path dataset tidak ditemukan: {train_dataset}")
       
    train_dataset = datasets.ImageFolder(train_path, train_transforms)
    val_dataset = datasets.ImageFolder(val_path, val_transform)
    print(f"📊 Train Dataset {args.dataset}: {len(train_dataset)} | Val samples: {len(val_dataset)}")
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,pin_memory=True)

    # 📈 Logging
    start_epoch = 0
    best_acc = 0
    history = []
    
    os.makedirs(args.output_dir, exist_ok=True)
    early_stop_counter = 0
    if os.path.exists(checkpoint_path):
        print(f"🔁 Auto-loading last checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        best_acc = checkpoint.get('best_acc', best_acc)
        start_epoch = checkpoint.get('epoch', 0)
        if os.path.exists(log_file):
            history = pd.read_csv(log_file).to_dict('records')

    print(f"✅ Train Model: {args.model}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        if epoch < 5:
            set_backbone_trainable(model, False)
        else:
            set_backbone_trainable(model, True)
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]", unit='batch')
        for batch_idx, (imgs, labels) in enumerate(loop):
            imgs, labels = imgs.to(device), labels.to(device)
            
            if np.random.rand() < 0.3:
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
                # Fix parsing output
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                    entropy_loss = outputs.get('entropy_loss', None)
                    gate_scores = outputs.get('gate_scores', None)
                elif isinstance(outputs, (tuple, list)):
                    logits, entropy_loss, gate_scores = outputs
                else:
                    logits = outputs
                    entropy_loss = None
                    gate_scores = None
                # Loss Mixup
                loss_cls = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)

                # Combine Loss with Entropy Regularization
                if entropy_loss is not None:
                    if gate_scores is not None:
                        lambda_entropy = entropy_controller.get_lambda(epoch, gate_scores)
                    else:
                        lambda_entropy = entropy_controller.get_lambda(epoch)
                    loss = loss_cls + lambda_entropy * (1 - entropy_loss)
                else:
                    loss = loss_cls
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
            
        train_acc = 100. * correct / total
        # 🔍 Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]", unit="batch")
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(val_loader_tqdm):
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(imgs)
                    # Fix parsing output
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                        entropy_loss = outputs.get('entropy_loss', None)
                        gate_scores = outputs.get('gate_scores', None)
                    elif isinstance(outputs, (tuple, list)):
                        logits, entropy_loss, gate_scores = outputs
                    else:
                        logits = outputs
                        entropy_loss = None
                        gate_scores = None

                    loss_cls = criterion(logits, labels)

                    # Combine Loss with Entropy Regularization
                    if entropy_loss is not None:
                        if gate_scores is not None:
                            lambda_entropy = entropy_controller.get_lambda(epoch, gate_scores)
                        else:
                            lambda_entropy = entropy_controller.get_lambda(epoch)
                        loss = loss_cls + lambda_entropy * (1 - entropy_loss)
                    else:
                        loss = loss_cls
                        
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                
                val_total += labels.size(0)
                val_loader_tqdm.set_postfix(val_loss=f"{loss.item():.4f}", acc=100.*val_correct/val_total)
                
        val_acc = 100 * val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1}: Loss:{loss_cls.item():.4f} | Entropy:{lambda_entropy:.4f} | gate score:{gate_scores} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss_avg:.4f}")
    
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_acc': best_acc
        }, checkpoint_path)
        
        # 💾 Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), base_path)
            print(f"✅ Model Terbaik di Simpan Acc: {best_acc:.2f}%")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"⏹️ Early stopping {early_stop_counter} dari {args.early_stop}")
            if early_stop_counter >= args.early_stop:
                print("⏹️ Early stopping triggered.")
                break
            
        # ⏺️ Logging CSV
        row = {"epoch": epoch+1, "loss": loss_cls, "entropy Loss": lambda_entropy, 
               "train_loss": running_loss / len(train_loader),
                "val_loss": val_loss_avg, "val_loss_acc": val_loss,"val_acc": val_acc}
        history.append(row)
        pd.DataFrame(history).to_csv(log_file, index=False)
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_acc': best_acc
        }, checkpoint_path)
        

# ⛳ Entry Point
if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='RAF-DB')
    parser.add_argument('--data_dir', type=str, default='../dataset_cropped_yolo/')
    parser.add_argument("--output_dir", type=str, default="logs")
    parser.add_argument("--backbone_weights", type=str, default="./weights/efficientvit_vggface2_best.pth")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument('--model', type=str, default='ddga', choices=['ddga','fusion','drm'])
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'])
    args = parser.parse_args()

    train(args)
