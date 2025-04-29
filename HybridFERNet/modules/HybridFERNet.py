import torch
import torch.nn as nn
import torch.nn.functional as F

# Import semua komponen dari modules/module.py
from modules.moduls import (
    EarlyFeatureExtractor,
    LightDPRA,  # Ganti dari DPRA_Enhanced menjadi LightDPRA
    AdaptiveContextPooling,
    UniversalFusion,
    DynamicRecalibration,
    SE_Block,
    LightResidualBlock,
)

class HybridFERNet(nn.Module):
    def __init__(self, num_classes=7, in_channels=3, 
                 fusion_mode='cross_gate',
                 use_residual=True, use_dynamic=True, use_se=True):
        super(HybridFERNet, self).__init__()
        self.use_residual = use_residual
        self.use_dynamic = use_dynamic
        self.use_se = use_se

        # Early feature extractor
        self.early_feat = EarlyFeatureExtractor(in_channels=in_channels, out_channels=64)
        self.feature_dim = 64

        # Pooling untuk menurunkan spatial size sebelum Local Branch
        self.pool_before_local = nn.MaxPool2d(kernel_size=2, stride=2)

        # Local and Global branches
        self.local_branch = LightDPRA(embed_dim=self.feature_dim)  # Pakai LightDPRA sekarang
        self.global_branch = AdaptiveContextPooling(dim=self.feature_dim)

        # Fusion Module
        self.fusion = UniversalFusion(feature_dim=self.feature_dim, fusion_mode=fusion_mode)

        # Optional Light Residual Enhancement after fusion
        if self.use_residual:
            self.residual_block = LightResidualBlock(self.fusion.out_channels)

        # Optional Post-Fusion Enhancement
        if self.use_dynamic:
            self.drm = DynamicRecalibration(channels=self.fusion.out_channels)
        if self.use_se:
            self.se_fusion = SE_Block(in_channels=self.fusion.out_channels)

        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.fusion.out_channels, num_classes)

    def forward(self, x):
        # Early shallow feature extraction
        feat = self.early_feat(x)
        feat_pooled = self.pool_before_local(feat)
        B, C, H, W = feat_pooled.size()

        # Local Pathway
        pra_feat = self.local_branch(feat_pooled)

        # Global Pathway
        app_feat = self.global_branch(feat)

        # Align spatial size if necessary
        if pra_feat.size()[2:] != app_feat.size()[2:]:
            app_feat = F.interpolate(app_feat, size=pra_feat.shape[2:], mode='bilinear', align_corners=False)

        # Fusion
        fused_feat = self.fusion(pra_feat, app_feat)

        # Optional Light Residual after Fusion
        if self.use_residual:
            fused_feat = self.residual_block(fused_feat)

        # Optional Dynamic Recalibration and SE Block
        if self.use_dynamic:
            fused_feat = self.drm(fused_feat)
        if self.use_se:
            fused_feat = self.se_fusion(fused_feat)

        # Pooling and Classification
        pooled = self.global_pool(fused_feat).flatten(1)
        out = self.classifier(self.dropout(pooled))
        return out
