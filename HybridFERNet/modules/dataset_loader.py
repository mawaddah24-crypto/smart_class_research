# modules/dataset_loader.py

import os
from torchvision import datasets
from torchvision import transforms

def dataset_loader(config, train_transform, val_transform):
    """
    Loader minimal untuk dataset RAF-DB atau FER2013
    Harus ada config['dataset']['train_dir'] dan config['dataset']['val_dir']
    """
    dataset_name = config['dataset']['name'].lower()

    if dataset_name in ['raf-db', 'rafdb']:
        train_dataset = datasets.ImageFolder(root=config['dataset']['train_dir'], transform=train_transform)
        val_dataset = datasets.ImageFolder(root=config['dataset']['val_dir'], transform=val_transform)

    elif dataset_name == 'fer2013':
        train_dataset = datasets.ImageFolder(root=config['dataset']['train_dir'], transform=train_transform)
        val_dataset = datasets.ImageFolder(root=config['dataset']['val_dir'], transform=val_transform)

    else:
        raise ValueError(f"Dataset {dataset_name} not supported yet!")

    return train_dataset, val_dataset
