import torch
import torch.nn as nn
import torch.nn.functional as F

# === Early Feature Extractor (Shallow Conv Stack) ===
class EarlyFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
# === Light Residual Block (optional) ===
class LightResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.relu(x + self.block(x))
    
# Positional Encoding untuk sequence
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=65536):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, embed_dim]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Auto-Adaptive SE Local block
class AutoAdaptiveSELocal(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        grid_size = 2 if min(h, w) <= 7 else 4

        avg_pool = F.adaptive_avg_pool2d(x, grid_size)
        se = self.fc1(avg_pool)
        se = self.relu(se)
        se = self.fc2(se)
        se = torch.sigmoid(se)
        se = F.interpolate(se, size=(h, w), mode='bilinear', align_corners=False)
        return x * se

# --- DPRA Enhanced with Positional Encoding---
class DPRA_Enhanced(nn.Module):
    def __init__(self, embed_dim, masking_ratio=0.5, use_masking=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.masking_ratio = masking_ratio
        self.use_masking = use_masking
        
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.se_local = AutoAdaptiveSELocal(in_channels=embed_dim)

    def forward(self, x):
        B, C, H, W = x.size()
        seq_len = H * W

        # Prepare sequence
        x = x.view(B, C, seq_len).transpose(1, 2)  # [B, Seq, C]
        x = self.positional_encoding(x)

        if self.use_masking:
            energy = x.mean(dim=-1)  # [B, Seq]
            k = max(1, int(seq_len * self.masking_ratio))
            topk_indices = torch.topk(energy, k=k, dim=1).indices
            mask = torch.zeros_like(energy)
            mask.scatter_(1, topk_indices, 1)
            mask = mask.unsqueeze(-1)
            x = x * mask

        # Multihead Attention
        attention_output, _ = self.attention(x, x, x)
        output = self.layer_norm(attention_output + x)
        output = self.fc(output)
        output = self.dropout(output)

        # Reshape to spatial map
        output = output.transpose(1, 2).view(B, C, H, W)

        # Apply SE local enhancement
        output = self.se_local(output)

        return output

# === Lightweight DPRA Local Attention Module ===
class LightDPRA(nn.Module):
    def __init__(self, embed_dim, masking_ratio=0.5, use_masking=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.masking_ratio = masking_ratio
        self.use_masking = use_masking

        self.conv_local = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.fc = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.size()

        if self.use_masking:
            energy = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
            k = max(1, int(H * W * self.masking_ratio))
            energy_flat = energy.view(B, -1)
            topk_indices = torch.topk(energy_flat, k=k, dim=1).indices

            mask = torch.zeros_like(energy_flat)
            mask.scatter_(1, topk_indices, 1)
            mask = mask.view(B, 1, H, W)
            x = x * mask

        x_local = self.conv_local(x)
        x_out = self.fc(x_local)
        x_out = self.dropout(x_out)

        return x_out
    
# --- FastSE ---
class FastSE(nn.Module):
    """
    Lightweight Channel Attention (Fast-SE).
    Global Average Pooling + Conv1x1 + Sigmoid for efficient channel recalibration.
    """
    def __init__(self, channels):
        super(FastSE, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = torch.mean(x, dim=(2, 3), keepdim=True)  # Global Average Pooling
        se = self.conv(se)
        se = self.sigmoid(se)
        return x * se

# --- AdaptiveContextPooling with FastSE ---
class AdaptiveContextPooling(nn.Module):
    """
    Adaptive Context Pooling (ACP) Module.
    Combines saliency-aware spatial attention, residual modulation, adaptive pooling,
    and lightweight channel attention (Fast-SE).
    """
    def __init__(self, dim, use_residual=True):
        super(AdaptiveContextPooling, self).__init__()
        self.use_residual = use_residual

        # Local saliency estimation
        self.saliency_conv = nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

        # Adaptive spatial pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        # Fast-SE for lightweight channel attention
        self.fast_se = FastSE(channels=dim)

    def forward(self, x):
        # Saliency map generation
        saliency = self.sigmoid(self.saliency_conv(x))

        # Apply saliency modulation (residual or pure)
        if self.use_residual:
            x = x * saliency + x  # Residual connection to preserve base feature
        else:
            x = x * saliency  # Pure modulation without residual

        # Adaptive pooling to 2x2 spatial size
        x = self.adaptive_pool(x)

        # Channel reweighting using Fast-SE
        x = self.fast_se(x)

        return x

# ✨ Dynamic Recalibration Module (DRM)
class DynamicRecalibration(nn.Module):
    def __init__(self, channels):
        super(DynamicRecalibration, self).__init__()
        self.depthwise_conv = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = self.depthwise_conv(x)
        se = self.sigmoid(se)
        return x * se + x  # residual connection

# === Squeeze-Excitation Channel Attention ===
class SE_Block(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.se(x).view(b, c, 1, 1)
        return x * y.expand_as(x)

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
