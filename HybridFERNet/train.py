# train.py - HybridFERNet Full Fitur Final Updated

import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from modules.HybridFERNet import HybridFERNet
from modules.moduls import dataset_loader  # Dataset loader sederhana
from modules.utils import accuracy, CrossEntropyLoss, get_scheduler
from modules.auhmentation import CutMix, MixUp, RandomErasing
from modules.utils import get_scheduler

from torch.cuda.amp import GradScaler, autocast


def parse_args():
    parser = argparse.ArgumentParser(description="Training HybridFERNet")
    parser.add_argument('--model_type', type=str, default='base', choices=['base', 'small', 'medium', 'large'])
    parser.add_argument('--dataset', type=str, default='RAF-DB', choices=['RAF-DB', 'FER2013', 'AffectNet', 'FERPlus'])
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['cosine', 'step', 'multistep', 'plateau', 'none'])
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--amp', action='store_true', help="Use Automatic Mixed Precision")
    return parser.parse_args()


def setup_config(args):
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['model_type'] = args.model_type
    config['dataset']['name'] = args.dataset
    config['scheduler'] = args.scheduler
    config['optimizer'] = args.optimizer
    config['use_amp'] = args.amp
    return config


def create_dirs(model_type):
    os.makedirs(f'checkpoints/{model_type}', exist_ok=True)
    os.makedirs(f'logs/', exist_ok=True)


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Augmentation Setup
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['input_size']) if config['augmentation'].get('random_resized_crop', True) else transforms.Resize((config['input_size'], config['input_size'])),
        transforms.RandomHorizontalFlip() if config['augmentation'].get('random_horizontal_flip', True) else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(config['augmentation'].get('random_rotation', 15)) if config['augmentation'].get('random_rotation', 0) > 0 else transforms.Lambda(lambda x: x),
        transforms.ColorJitter(brightness=0.2, contrast=0.2) if config['augmentation'].get('color_jitter', True) else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if config['augmentation'].get('random_erasing', False):
        train_transform.transforms.append(RandomErasing())

    val_transform = transforms.Compose([
        transforms.Resize((config['input_size'], config['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset, val_dataset = dataset_loader(config, train_transform, val_transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    model = HybridFERNet(num_classes=config['num_classes']).to(device)

    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])

    scheduler = get_scheduler(optimizer, config)
    criterion = CrossEntropyLoss()
    scaler = GradScaler(enabled=config['use_amp'])

    cutmix = CutMix(prob=config['augmentation'].get('use_cutmix', False))
    mixup = MixUp(prob=config['augmentation'].get('use_mixup', False))

    best_val_acc = 0
    patience_counter = 0
    patience = config.get('early_stopping_patience', 10)

    checkpoint_path = f'checkpoints/{config['model_type']}/last.pt'
    best_path = f'checkpoints/{config['model_type']}/best.pt'
    log_path = f'logs/{config['model_type']}_training_log.csv'

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']

    if not os.path.exists(log_path):
        pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr', 'train_time', 'eval_time']).to_csv(log_path, index=False)

    for epoch in range(start_epoch, config['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_train_time = time.time()

        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if config['augmentation'].get('use_cutmix', False):
                inputs, targets = cutmix(inputs, labels)
            elif config['augmentation'].get('use_mixup', False):
                inputs, targets = mixup(inputs, labels)
            else:
                targets = labels

            with autocast(enabled=config['use_amp']):
                outputs = model(inputs)

                if isinstance(targets, tuple):  # For CutMix or MixUp
                    targets_a, targets_b, lam = targets
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            train_bar.set_postfix(loss=running_loss/total, acc=correct/total)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_time = time.time() - start_train_time

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        start_eval_time = time.time()

        val_bar = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast(enabled=config['use_amp']):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

                val_bar.set_postfix(loss=val_loss/val_total, acc=val_correct/val_total)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        eval_time = time.time() - start_eval_time

        if scheduler:
            if config['scheduler'] == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        log_row = pd.DataFrame([[epoch+1, train_loss, train_acc, val_loss, val_acc, current_lr, train_time, eval_time]], columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr', 'train_time', 'eval_time'])
        log_row.to_csv(log_path, mode='a', header=False, index=False)

        print(f"Epoch {epoch+1}: Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

        save_dict = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'best_val_acc': best_val_acc
        }
        torch.save(save_dict, checkpoint_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(save_dict, best_path)
            print(f"Best model updated at epoch {epoch+1} with Val Acc {best_val_acc:.4f}")

        if val_acc <= best_val_acc:
            patience_counter += 1
        else:
            patience_counter = 0

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training Completed!")


if __name__ == '__main__':
    args = parse_args()
    config = setup_config(args)
    create_dirs(config['model_type'])
    train(config)
