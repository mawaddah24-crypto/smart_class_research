import torch
import torch.nn as nn
from module import (PRA, 
                    AdaptivePatchPooling, 
                    DSE_Local, DSE_Global, SE_Local, SE_Global, DDGA, SE_Block,
                    SEBlock_Enhanced, 
                    PartialAttentionMasking)
from timm import create_model

class DualPath_AblationBase(nn.Module):
    def __init__(self, 
                 num_classes=7, 
                 pretrained=True,
                 use_pa=False,
                 use_dse=False,
                 use_ddga=False,
                 masking_ratio=0.5):
        
        super(DualPath_AblationBase, self).__init__()

        # === Backbone ===
        self.backbone = create_model("efficientvit_b1.r224_in1k", pretrained=pretrained, features_only=True, in_chans=3)
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']

        # === Partial Attention (Optional) ===
        self.use_pa = use_pa
        if self.use_pa:
            self.partial_attention = PartialAttentionMasking(masking_ratio=masking_ratio)

        # === Local and Global Pathways ===
        self.pra = PRA(in_channels=self.feature_dim)
        self.app = AdaptivePatchPooling(in_channels=self.feature_dim)

        if use_dse:
            self.se_local = DSE_Local(in_channels=self.feature_dim)
            self.se_global = DSE_Global(in_channels=self.feature_dim)
        else:
            self.se_local = SE_Local(in_channels=self.feature_dim)
            self.se_global = SE_Global(in_channels=self.feature_dim)

        # === Fusion ===
        self.use_ddga = use_ddga
        if self.use_ddga:
            self.ddga = DDGA(in_channels=self.feature_dim)
        else:
            self.fc_fusion = nn.Sequential(
                nn.Conv2d(self.feature_dim * 2, self.feature_dim, kernel_size=1),
                nn.BatchNorm2d(self.feature_dim),
                nn.ReLU(inplace=True)
            )

        self.se_fusion = SE_Block(self.feature_dim)

        # === Classifier ===
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        B = x.size(0)

        # Backbone feature
        feat = self.backbone(x)[-1]  # Output [B, C, H, W]

        # Apply Partial Attention if enabled
        if self.use_pa:
            feat = self.partial_attention(feat)

        # Local and Global Pathways
        pra_feat = self.pra(feat).transpose(1, 2).view(feat.shape)
        pra_feat = self.se_local(pra_feat)

        app_feat = self.app(feat)
        app_feat = self.se_global(app_feat)

        # Fusion
        if self.use_ddga:
            fused_feat = self.ddga(pra_feat, app_feat)
        else:
            fused_feat = torch.cat([pra_feat, app_feat], dim=1)
            fused_feat = self.fc_fusion(fused_feat)

        fused_feat = self.se_fusion(fused_feat)

        # Classification
        pooled = self.global_pool(fused_feat).view(B, -1)
        out = self.classifier(self.dropout(pooled))

        return out
