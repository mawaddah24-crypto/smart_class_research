# augmentations.py - Modular Augmentation for HybridFERNet

import torch
import random
from torchvision import transforms
import numpy as np

class CutMix:
    def __init__(self, beta=1.0, prob=0.5):
        self.beta = beta
        self.prob = prob

    def __call__(self, inputs, targets):
        if random.random() > self.prob:
            return inputs, targets

        lam = np.random.beta(self.beta, self.beta)
        batch_size = inputs.size(0)
        index = torch.randperm(batch_size)

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(inputs.size(), lam)
        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[index, :, bbx1:bbx2, bby1:bby2]

        targets_a, targets_b = targets, targets[index]
        return inputs, (targets_a, targets_b, lam)

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

class RandomErasing:
    def __init__(self, p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)):
        self.eraser = transforms.RandomErasing(p=p, scale=scale, ratio=ratio, value='random')

    def __call__(self, img):
        return self.eraser(img)

class MixUp:
    def __init__(self, alpha=0.4, prob=0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, inputs, targets):
        if random.random() > self.prob:
            return inputs, targets

        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = inputs.size(0)
        index = torch.randperm(batch_size)

        mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
        targets_a, targets_b = targets, targets[index]
        return mixed_inputs, (targets_a, targets_b, lam)

# Standard train transforms

def get_standard_augmentation(config):
    transform_list = []

    if config['augmentation'].get('random_resized_crop', True):
        transform_list.append(transforms.RandomResizedCrop(config['input_size']))
    else:
        transform_list.append(transforms.Resize((config['input_size'], config['input_size'])))

    if config['augmentation'].get('random_horizontal_flip', True):
        transform_list.append(transforms.RandomHorizontalFlip())

    if config['augmentation'].get('random_rotation', 0) > 0:
        transform_list.append(transforms.RandomRotation(config['augmentation']['random_rotation']))

    if config['augmentation'].get('color_jitter', True):
        transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    if config['augmentation'].get('random_erasing', False):
        transform_list.append(RandomErasing(p=0.5))

    return transforms.Compose(transform_list)

# Validation transforms

def get_validation_augmentation(config):
    return transforms.Compose([
        transforms.Resize((config['input_size'], config['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
