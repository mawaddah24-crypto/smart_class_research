import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
import os, csv
from modules.moduls import LightDPRA, AdaptiveContextPooling, UniversalFusion, SE_Block, DynamicRecalibration

def log_feature(self, name, tensor):
    if self.verbose:
        print(f"[{name}] Shape: {tensor.shape}, Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}")

# ==== Simple Adaptive Pooling (for ablation, no saliency) ====
class SimpleGlobalPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x):
        return self.pool(x)
    
# ==== Simple Sum Fusion (for ablation) ====
class SimpleSumFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, local_feat, global_feat):
        return local_feat + global_feat


# ==== Base DualPathModel Fusion ====
class BaselineFinal(nn.Module):
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', pretrained=True, 
                 in_channels=3, fusion_mode='confidence', use_drm=False, use_se=False):
        """
        BaselineFinal:
        - Dual Pathway: Local (DPRA_Enhanced) + Global (AdaptiveContextPooling)
        - Fusion via UniversalFusion (supports cross_gate or confidence)
        - Optional post-fusion Dynamic Recalibration and SE Block
        
        Args:
            num_classes: output class number
            backbone_name: backbone network (default efficientvit_b1)
            pretrained: use pretrained backbone
            in_channels: input channel (default RGB 3)
            fusion_mode: 'cross_gate' or 'confidence'
            use_drm: use Dynamic Recalibration Module after fusion
            use_se: use SE Block after fusion
        """
        super(BaselineFinal, self).__init__()
        self.use_drm = use_drm
        self.use_se = use_se

        # Backbone feature extractor
        self.backbone = create_model(backbone_name, pretrained=pretrained, features_only=True, in_chans=in_channels)
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']

        # Local and Global Branch
        self.local_branch = LightDPRA(embed_dim=self.feature_dim)
        self.global_branch = AdaptiveContextPooling(dim=self.feature_dim)

        # Fusion Module
        self.fusion = UniversalFusion(feature_dim=self.feature_dim, fusion_mode=fusion_mode)

        # Post-Fusion Enhancement Modules
        if use_drm:
            self.drm = DynamicRecalibration(channels=self.fusion.out_channels)
            
        if use_se:
            self.se_fusion = SE_Block(in_channels=self.fusion.out_channels)

        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.fusion.out_channels, num_classes)

    def forward(self, x):
        feat = self.backbone(x)[-1]  # Output: [B, C, H, W]
        B, C, H, W = feat.size()

        # Local Pathway (DPRA Enhanced)
        pra_feat = self.local_branch(feat)
        pra_feat = pra_feat.transpose(1, 2).reshape(B, C, H, W)
        #pra_feat = pra_feat.transpose(1, 2).view(B, C, H, W)

        # Global Pathway (Adaptive Context Pooling)
        app_feat = self.global_branch(feat)

        # Align spatial size if necessary
        if pra_feat.size()[2:] != app_feat.size()[2:]:
            app_feat = F.interpolate(app_feat, size=pra_feat.shape[2:], mode='bilinear', align_corners=False)

        # Fusion
        fused_feat = self.fusion(pra_feat, app_feat)

        # Post-Fusion Dynamic Recalibration
        if self.use_drm:
            fused_feat = self.drm(fused_feat)

        # Post-Fusion SE Block
        if self.use_se:
            fused_feat = self.se_fusion(fused_feat)

        # Pooling and Classification
        pooled = self.global_pool(fused_feat).flatten(1)
        out = self.classifier(self.dropout(pooled))

        return out
                    