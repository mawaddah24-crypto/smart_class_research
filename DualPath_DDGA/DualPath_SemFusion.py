import torch
import torch.nn as nn
from timm import create_model

from module import (
    SemanticAwarePartialAttention,
    DynamicGridDSE,
    CrossLevelDualGatedAttention,
    PRA_MultiheadAttention,
    AGAPP,
    SEBlock_Enhanced
)

class DualPath_SemFusion(nn.Module):
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', pretrained=True, 
                 in_channels=3, masking_ratio=0.5):
        super(DualPath_SemFusion, self).__init__()
        
        # === Semantic-Aware Partial Attention ===
        self.semantic_attention = SemanticAwarePartialAttention(masking_ratio=masking_ratio)
        
        # === Backbone ===
        self.backbone = create_model(backbone_name, pretrained=pretrained, features_only=True, in_chans=in_channels)
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']  # biasanya 128

        # === Local Pathway ===
        self.local_branch = PRA_MultiheadAttention(embed_dim=self.feature_dim)
        self.local_refine = DynamicGridDSE(in_channels=self.feature_dim, out_channels=self.feature_dim)

        # === Global Pathway ===
        self.global_branch = AGAPP(dim=self.feature_dim)
        self.global_refine = DynamicGridDSE(in_channels=self.feature_dim, out_channels=self.feature_dim)

        # === Fusion Pathway ===
        self.cross_level_fusion = CrossLevelDualGatedAttention(dim=self.feature_dim)

        # === Final Attention and Classifier ===
        self.se_fusion = SEBlock_Enhanced(in_channels=self.feature_dim)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)[-1]  # [B, C, H, W]
        B, C, H, W = feat.shape

        # === Apply Semantic Partial Attention ===
        feat = self.semantic_attention(feat)

        # === Local Pathway ===
        local_feat = self.local_branch(feat)                   # [B, H*W, C]
        local_feat = local_feat.transpose(1, 2).view(B, C, H, W)
        local_feat = self.local_refine(local_feat)

        # === Global Pathway ===
        global_feat = self.global_branch(feat)                  # [B, C, H, W]
        global_feat = self.global_refine(global_feat)

        # === Cross-Level Fusion ===
        fused_feat = self.cross_level_fusion(local_feat, global_feat)

        # === Final Attention ===
        fused_feat = self.se_fusion(fused_feat)

        # === Classification Head ===
        pooled = self.global_pool(fused_feat).view(B, C)        # [B, C]
        out = self.classifier(self.dropout(pooled))             # [B, num_classes]

        return out
