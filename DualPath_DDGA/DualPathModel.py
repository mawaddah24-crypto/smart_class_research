import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from module import (
    PartialAttentionMasking, AdvancedPartialAttentionMasking, SemanticAwarePartialAttention,
    DSE_Global, DynamicGridDSE, DynamicGridDSEPlus,
    DSE_Local, ConfidenceAwareFusion,ConfidenceAwareFusionV2,  DynamicRecalibration,
    DDGA, DDGA_DSF, FeatureAwareDDGA, AdaptiveContextAPP,
    AGAPP, 
    PRA_MultiheadAttention, D_PRA,
    PRA, SE_Global, SE_Local, SE_Block, AdaptivePatchPooling, SemanticAwarePooling)


# ContextResidualDRM (CR-DRM)
class ContextResidualDRM(nn.Module):
    def __init__(self, feature_dim, reduction=8):
        super(ContextResidualDRM, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // reduction, feature_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        context = self.avg_pool(x).view(b, c)  # Global feature descriptor
        scaling = self.fc(context).view(b, c, 1, 1)
        out = x + x * scaling  # Residual Recalibration (not replace x, but refine)
        return out


#SoftSEBlock 
class SoftSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SoftSEBlock, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x + x * y  # Residual Gating style (soft enhancement)
        return out

class DualPath_Baseline(nn.Module):
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', pretrained=True, 
                 in_channels=3,delay_ddga=True):
        super(DualPath_Baseline, self).__init__()
        self.delay_ddga = delay_ddga  # True: delay DDGA activation
        # Backbone: EfficientViT pretrained
        self.backbone = create_model(backbone_name, pretrained=pretrained, features_only=True, in_chans=in_channels)
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']  # usually 128

        # Partial Attention Masking
        #self.partial_attention = SemanticAwarePartialAttention()  # atau SAPA jika sudah
        #self.dynamic_grid_dse =  DynamicGridDSE(in_channels=self.feature_dim,)
        # Local and Global Pathways
        self.pra = D_PRA(embed_dim=self.feature_dim)
        self.app = AdaptiveContextAPP(dim=self.feature_dim)

        # Dynamic Spatial Excitation (Local & Global)
        # Base
        #self.dse_local = SE_Local(in_channels=self.feature_dim)
        #self.dse_global = SE_Global(in_channels=self.feature_dim)

        # Model v1
        self.dse_local = DSE_Local(in_channels=self.feature_dim, out_channels=self.feature_dim)
        self.dse_global = DSE_Global(in_channels=self.feature_dim, out_channels=self.feature_dim)
        
        # Model v3
        #self.dse_local = DynamicGridDSE(in_channels=self.feature_dim, out_channels=self.feature_dim)
        #self.dse_global = DynamicGridDSE(in_channels=self.feature_dim, out_channels=self.feature_dim)
        
        self.simple_fusion = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1)
        # Dynamic attention to fuse both pathways
        self.ddga = DDGA(dim=self.feature_dim)
            
        # Semantic-Aware Pooling (NEW)
        self.semantic_pool = SE_Block(in_channels=self.feature_dim)

         # Final pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Dropout and Classifier
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # Backbone
        feat = self.backbone(x)[-1]  # Feature map [B, C, H, W]
        B, C, H, W = feat.size()
        # Partial Attention
        #feat = self.partial_attention(feat)
        #feat = self.dynamic_grid_dse(feat)
        
        # Local Path
        pra_feat = self.pra(feat).transpose(1, 2).view(feat.shape)
        pra_feat = self.dse_local(pra_feat)

        # Global Path
        app_feat = self.app(feat)
        app_feat = self.dse_global(app_feat)

        # Fusion
        if pra_feat.size()[2:] != app_feat.size()[2:]:
            app_feat = F.interpolate(app_feat, size=pra_feat.shape[2:], mode='bilinear', align_corners=False)
            
        # === Dual Dynamic Gated Attention Fusion ===
        if self.delay_ddga:
            # Stage 2: adaptive fusion with DDGA
            fused_feat = self.ddga(pra_feat, app_feat)
        else:
            # Stage 1: simple fusion
            fused_feat = pra_feat + app_feat
            fused_feat = self.simple_fusion(fused_feat)
            entropy = None

        # Semantic-Aware Pooling
        fused_feat = self.semantic_pool(fused_feat)
        pooled = self.global_pool(fused_feat).view(B, C)  # [B, C]
             
        # Classification
        out = self.classifier(self.dropout(pooled))

        return out

class DualPath_Fusion(nn.Module):
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', pretrained=True, 
                 in_channels=3, warmup_epochs=5):
        super(DualPath_Fusion, self).__init__()

        # Backbone
        self.backbone = create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels
        )
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']

        # Local pathway
        self.pra = D_PRA(embed_dim=self.feature_dim)
        self.se_local = SE_Local(in_channels=self.feature_dim)

        # Global pathway
        self.app = AdaptiveContextAPP(dim=self.feature_dim)
        self.se_global = SE_Global(in_channels=self.feature_dim)

        # Confidence Aware Fusion V2
        self.conf_fusion = ConfidenceAwareFusion(feature_dim=self.feature_dim)

        #  Dynamic Recalibration Module (DRM)
        self.drm = DynamicRecalibration(channels=self.feature_dim)
        self.semantic_pool = SE_Block(in_channels=self.feature_dim)
        
        # Final Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)[-1]
        B, C, H, W = feat.size()

        # Local branch
        pra_feat = self.pra(feat).transpose(1, 2).view(B, C, H, W)
        pra_feat = self.se_local(pra_feat)

        # Global branch
        app_feat = self.app(feat)
        app_feat = self.se_global(app_feat)

        # Align spatial size
        if app_feat.shape[-2:] != pra_feat.shape[-2:]:
            app_feat = F.interpolate(app_feat, size=pra_feat.shape[2:], mode='bilinear', align_corners=False)

        # Confidence-aware fusion
        fused_feat, alpha, beta = self.conf_fusion(pra_feat, app_feat)

        #  Dynamic Recalibration Module (DRM)
        fused_feat = self.drm(fused_feat)

        # Semantic Pooling
        fused_feat = self.semantic_pool(fused_feat)

        # Global pooling and classification
        pooled = self.global_pool(fused_feat).view(B, C)
        out = self.classifier(self.dropout(pooled))

        return out

class DualPath_CR_DRM(nn.Module):
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', pretrained=True, 
                 in_channels=3, warmup_epochs=5):
        super(DualPath_CR_DRM, self).__init__()

        # Backbone
        self.backbone = create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels
        )
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']

        # Local pathway
        self.pra = D_PRA(embed_dim=self.feature_dim)
        self.se_local = SE_Local(in_channels=self.feature_dim)

        # Global pathway
        self.app = AdaptiveContextAPP(dim=self.feature_dim)
        self.se_global = SE_Global(in_channels=self.feature_dim)

        # Lightweight attention Confidence Aware Fusion V2
        self.conf_fusion = ConfidenceAwareFusionV2(feature_dim=self.feature_dim, warmup_epochs=warmup_epochs)

        # Context Residual DRM and Semantic Pooling
        self.drm = ContextResidualDRM(feature_dim=self.feature_dim)
        self.semantic_pool = SoftSEBlock(in_channels=self.feature_dim)
        
        # Final Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)[-1]
        B, C, H, W = feat.size()

        # Local branch
        pra_feat = self.pra(feat).transpose(1, 2).view(B, C, H, W)
        pra_feat = self.se_local(pra_feat)

        # Global branch
        app_feat = self.app(feat)
        app_feat = self.se_global(app_feat)

        # Align spatial size
        if app_feat.shape[-2:] != pra_feat.shape[-2:]:
            app_feat = F.interpolate(app_feat, size=pra_feat.shape[2:], mode='bilinear', align_corners=False)

        # Lightweight attention Confidence Aware Fusion V2
        fused_feat, alpha, beta = self.conf_fusion(pra_feat, app_feat)

        # Context Residual Recalibration (optional)
        fused_feat = self.drm(fused_feat)
        
        # Semantic Pooling
        fused_feat = self.semantic_pool(fused_feat)

        # Global pooling and classification
        pooled = self.global_pool(fused_feat).view(B, C)
        out = self.classifier(self.dropout(pooled))

        return out

    def update_confidence_epoch(self, epoch):
        """Method untuk update current epoch ke confidence fusion block."""
        self.conf_fusion.set_epoch(epoch)