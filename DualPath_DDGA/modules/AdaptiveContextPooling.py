import torch
import torch.nn as nn
import torch.nn.functional as F

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
