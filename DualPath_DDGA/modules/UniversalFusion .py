import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Cross-Level Dual Gated Attention Fusion ---
class CrossLevelDualGatedAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.global_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.cross_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion_proj = nn.Conv2d(dim * 2, dim * 2, kernel_size=1)  # final projection

    def forward(self, local_feat, global_feat):
        B, C, H, W = local_feat.shape

        local_proj = self.local_proj(local_feat)
        global_proj = self.global_proj(global_feat)

        if global_proj.shape[-2:] != (H, W):
            global_proj = F.interpolate(global_proj, size=(H, W), mode='bilinear', align_corners=False)

        concat_feat = torch.cat([local_proj, global_proj], dim=1)
        gate = self.cross_gate(concat_feat)

        gated_local = local_proj * gate
        gated_global = global_proj * (1 - gate)

        fused_feat = torch.cat([gated_local, gated_global], dim=1)
        out = self.fusion_proj(fused_feat)
        return out

# --- Confidence Aware Fusion ---
class ConfidenceAwareFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.local_fc = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.AdaptiveAvgPool2d(1)

    def forward(self, local_feat, global_feat):
        # Compute scalar confidence scores
        local_conf = self.local_fc(local_feat).view(local_feat.size(0), -1).mean(dim=1, keepdim=True)
        global_conf = self.global_fc(global_feat).view(global_feat.size(0), -1).mean(dim=1, keepdim=True)
        sum_conf = local_conf + global_conf + 1e-6  # Avoid divide by zero
        alpha = local_conf / sum_conf
        beta = global_conf / sum_conf

        # Weighted fusion
        fused = alpha.view(-1, 1, 1, 1) * local_feat + beta.view(-1, 1, 1, 1) * global_feat

        return fused

# --- Universal Fusion Wrapper ---
class UniversalFusion(nn.Module):
    """
    UniversalFusion:
    A flexible fusion module supporting multiple fusion strategies.
    Currently supports:
    - 'cross_gate' : CrossLevelDualGatedAttention
    - 'confidence' : ConfidenceAwareFusion
    """
    def __init__(self, feature_dim, fusion_mode='cross_gate'):
        super().__init__()
        self.fusion_mode = fusion_mode

        if fusion_mode == 'cross_gate':
            self.fusion = CrossLevelDualGatedAttention(dim=feature_dim)
            self.out_channels = feature_dim * 2  # fusion output is 2C
        elif fusion_mode == 'confidence':
            self.fusion = ConfidenceAwareFusion(feature_dim=feature_dim)
            self.out_channels = feature_dim  # fused feature has C channels
        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")

    def forward(self, local_feat, global_feat):
        if self.fusion_mode == 'cross_gate':
            fused = self.fusion(local_feat, global_feat)
            return fused
        elif self.fusion_mode == 'confidence':
            fused = self.fusion(local_feat, global_feat)
            return fused
