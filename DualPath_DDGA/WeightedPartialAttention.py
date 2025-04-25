# WeightedPartialAttention.py
import torch
import torch.nn as nn

class WeightedPartialAttention(nn.Module):
    def __init__(self, masking_ratio=0.5, alpha=0.6, beta=0.2, gamma=0.2):
        super(WeightedPartialAttention, self).__init__()
        self.masking_ratio = masking_ratio
        self.alpha = alpha  # weight for feature activation
        self.beta = beta    # weight for gaze importance
        self.gamma = gamma  # weight for pose importance

    def forward(self, x, gaze_importance, pose_importance):
        B, C, H, W = x.shape
        x_flat = x.flatten(2)  # [B, C, H*W]

        # Feature energy
        feature_energy = x_flat.norm(dim=1)  # [B, H*W]

        # Normalize gaze_importance and pose_importance to [0,1]
        gaze_importance = torch.sigmoid(gaze_importance)
        pose_importance = torch.sigmoid(pose_importance)

        # Combine weighted scores
        combined_score = (self.alpha * feature_energy +
                          self.beta * gaze_importance +
                          self.gamma * pose_importance)

        k = int(self.masking_ratio * combined_score.shape[1])
        topk_values, topk_indices = torch.topk(combined_score, k, dim=1)

        mask = torch.zeros_like(combined_score).to(x.device)
        mask.scatter_(1, topk_indices, 1.0)

        mask = mask.unsqueeze(1)  # [B, 1, H*W]
        mask = mask.expand(-1, C, -1)
        mask = mask.view(B, C, H, W)

        x = x * mask
        return x
