import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

class CASME2SequenceDataset(Dataset):
    def __init__(self, root_dir, seq_len=18, transform=None):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            for sample in os.listdir(cls_folder):
                sample_folder = os.path.join(cls_folder, sample)
                if os.path.isdir(sample_folder):
                    self.samples.append((sample_folder, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_folder, label = self.samples[idx]
        frame_files = sorted(os.listdir(sample_folder))

        frames = []
        for frame_name in frame_files:
            img_path = os.path.join(sample_folder, frame_name)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # Pad/Trim sequence
        if len(frames) < self.seq_len:
            # Repeat last frame
            frames.extend([frames[-1]] * (self.seq_len - len(frames)))
        else:
            frames = frames[:self.seq_len]

        frames = torch.stack(frames, dim=0)  # [seq_len, C, H, W]
        return frames, label
