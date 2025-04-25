# PartialAttentionMasking_Original.py
import torch
import torch.nn as nn

class PartialAttentionMasking(nn.Module):
    def __init__(self, masking_ratio=0.5, strategy='topk'):
        super(PartialAttentionMasking, self).__init__()
        self.masking_ratio = masking_ratio
        self.strategy = strategy

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2)  # [B, C, H*W]

        # Compute energy per patch
        energy = x_flat.norm(dim=1)  # [B, H*W]

        if self.strategy == 'topk':
            k = int(self.masking_ratio * energy.shape[1])
            topk_values, topk_indices = torch.topk(energy, k, dim=1)

            mask = torch.zeros_like(energy).to(x.device)
            mask.scatter_(1, topk_indices, 1.0)

            mask = mask.unsqueeze(1)  # [B, 1, H*W]
            mask = mask.expand(-1, C, -1)
            mask = mask.view(B, C, H, W)

            x = x * mask  # Masked feature map
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

        return x
