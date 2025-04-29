import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from module import (
    SemanticAwarePartialAttention,
    DynamicGridDSE,
    CrossLevelDualGatedAttention,
    PRA_MultiheadAttention,
    AGAPP,
    SE_Block
)

class SEBlock_Local(nn.Module):
    def __init__(self, in_channels, reduction=16, grid_size=2):
        super(SEBlock_Local, self).__init__()
        self.grid_pool = nn.AdaptiveAvgPool2d(grid_size)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = self.grid_pool(x)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)

        se = F.interpolate(se, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return x * se
    
class SEBlock_Global(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock_Global, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = torch.mean(x, dim=(2, 3), keepdim=True)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        return x * se

# ====== CrossLevelSEBlock ======
class CrossLevelSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, grid_size=2):
        super(CrossLevelSEBlock, self).__init__()
        assert in_channels % 2 == 0, "in_channels harus genap karena akan di-split."

        self.local_se = SEBlock_Local(in_channels // 2, grid_size=grid_size)
        self.global_se = SEBlock_Global(in_channels // 2)

    def forward(self, x):
        B, C, H, W = x.size()
        
        # Split hasil fusion ke Local dan Global
        local_feat, global_feat = torch.chunk(x, chunks=2, dim=1)

        # Apply masing-masing DSE
        local_feat = self.local_se(local_feat)
        global_feat = self.global_se(global_feat)

        # Combine kembali
        out = torch.cat([local_feat, global_feat], dim=1)

        return out


class DualPath_SemFusion(nn.Module):
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', pretrained=True, 
                 in_channels=3, masking_ratio=0.5):
        super(DualPath_SemFusion, self).__init__()
        
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
        self.se_fusion = CrossLevelSEBlock(in_channels=self.feature_dim * 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        #self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.classifier = nn.Linear(self.feature_dim * 2, num_classes)

    def forward(self, x):
        feat = self.backbone(x)[-1]  # [B, C, H, W]
        B, C, H, W = feat.shape

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
        pooled = self.global_pool(fused_feat)
        pooled = pooled.view(B, -1)
        out = self.classifier(self.dropout(pooled))
        
        return out
