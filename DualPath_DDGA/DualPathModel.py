import torch
import torch.nn as nn
from timm import create_model
from module import PartialAttentionMasking, DSE_Global, DSE_Local, DDGA,AGAPP, SEBlock_Enhanced, PRA_MultiheadAttention
from module import PRA, SE_Global, SE_Local, SE_Block, SemanticAwarePooling, AdaptivePatchPooling
                    

class DualPath_Baseline(nn.Module):
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', pretrained=True, in_channels=3):
        super(DualPath_Baseline, self).__init__()
        
        # Backbone: EfficientViT pretrained
        self.backbone = create_model(backbone_name, pretrained=pretrained, features_only=True, in_chans=in_channels)
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']  # usually 128

        # Local pathway
        self.pra = PRA(embed_dim=self.feature_dim)
        self.se_local = SE_Local(in_channels=self.feature_dim)

        # Global pathway
        self.app = AdaptivePatchPooling(dim=self.feature_dim)
        self.se_global = SE_Global(in_channels=self.feature_dim)

        # Dynamic attention to fuse both pathways
        self.ddga = DDGA(dim=self.feature_dim)

        # Final SE Fusion block
        self.se_fusion = SE_Block(in_channels=self.feature_dim)

        # Final pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)[-1]  # Extract deepest feature: [B, C, H, W]
        B, C, H, W = feat.size()

        # === Local Pathway ===
        pra_feat = self.pra(feat)                           # [B, H*W, C]
        pra_feat = pra_feat.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        pra_feat = self.se_local(pra_feat)

        # === Global Pathway ===
        app_feat = self.app(feat)
        app_feat = self.se_global(app_feat)

        # === Dual Dynamic Gated Attention Fusion ===
        fused_feat = self.ddga(pra_feat, app_feat)

        # === Final Fusion Attention via SE ===
        fused_feat = self.se_fusion(fused_feat)

        # === Classification Head ===
        pooled = self.global_pool(fused_feat).view(B, C)  # [B, C]
        out = self.classifier(self.dropout(pooled))       # [B, num_classes]

        return out

class DualPath_Baseline_DSE(nn.Module):
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', pretrained=True, in_channels=3):
        super(DualPath_Baseline, self).__init__()
        
        # Backbone: EfficientViT pretrained
        self.backbone = create_model(backbone_name, pretrained=pretrained, features_only=True, in_chans=in_channels)
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']  # usually 128

        # Local pathway
        self.pra = PRA(embed_dim=self.feature_dim)
        self.se_local = SE_Local(in_channels=self.feature_dim)

        # Global pathway
        self.app = AdaptivePatchPooling(dim=self.feature_dim)
        self.se_global = SE_Global(in_channels=self.feature_dim)

        # Dynamic attention to fuse both pathways
        self.ddga = DDGA(dim=self.feature_dim)

        # Final SE Fusion block
        self.se_fusion = SE_Block(in_channels=self.feature_dim)

        # Final pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)[-1]  # Extract deepest feature: [B, C, H, W]
        B, C, H, W = feat.size()

        # === Local Pathway ===
        pra_feat = self.pra(feat)                           # [B, H*W, C]
        pra_feat = pra_feat.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        pra_feat = self.se_local(pra_feat)

        # === Global Pathway ===
        app_feat = self.app(feat)
        app_feat = self.se_global(app_feat)

        # === Dual Dynamic Gated Attention Fusion ===
        fused_feat = self.ddga(pra_feat, app_feat)

        # === Final Fusion Attention via SE ===
        fused_feat = self.se_fusion(fused_feat)

        # === Classification Head ===
        pooled = self.global_pool(fused_feat).view(B, C)  # [B, C]
        out = self.classifier(self.dropout(pooled))       # [B, num_classes]

        return out
    
class DualPath_PartialAttentionModif(nn.Module):
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', pretrained=True, in_channels=3):
        super(DualPath_PartialAttentionModif, self).__init__()
        
       
        # Backbone: EfficientViT pretrained
        self.backbone = create_model(backbone_name, pretrained=pretrained, features_only=True, in_chans=in_channels)
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']  # usually 128
        
        self.partial_attention = PartialAttentionMasking(masking_ratio=0.5, strategy="topk")
        # Local pathway
        self.pra = PRA_MultiheadAttention(embed_dim=self.feature_dim)
        self.se_local = DSE_Local(in_channels=self.feature_dim, out_channels=self.feature_dim)

        # Global pathway
        self.app = AGAPP(dim=self.feature_dim)
        self.se_global = DSE_Global(in_channels=self.feature_dim, out_channels=self.feature_dim)

        # Dynamic attention to fuse both pathways
        self.ddga = DDGA(dim=self.feature_dim)

        # Final SE Fusion block
        self.se_fusion = SEBlock_Enhanced(in_channels=self.feature_dim)

        # Final pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)[-1]  # Extract deepest feature: [B, C, H, W]
        B, C, H, W = feat.size()

        # Apply Partial Attention after feature extraction
        feat = self.partial_attention(feat)

        # === Local Pathway ===
        pra_feat = self.pra(feat)
        pra_feat = pra_feat.transpose(1, 2).view(B, C, H, W)
        pra_feat = self.se_local(pra_feat)

        # === Global Pathway ===
        app_feat = self.app(feat)
        app_feat = self.se_global(app_feat)

        # === Fusion ===
        fused_feat = self.ddga(pra_feat, app_feat)
        fused_feat = self.se_fusion(fused_feat)

        # === Classification Head ===
        pooled = self.global_pool(fused_feat)
        pooled = pooled.view(B, pooled.size(1))
        out = self.classifier(self.dropout(pooled))

        return out

class DualPath_Base_Partial(nn.Module):
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', pretrained=True, in_channels=3):
        super(DualPath_Base_Partial, self).__init__()
        
        self.partial_attention = PartialAttentionMasking(masking_ratio=0.5, strategy="topk")
        # Backbone: EfficientViT pretrained
        self.backbone = create_model(backbone_name, pretrained=pretrained, features_only=True, in_chans=in_channels)
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']  # usually 128

        # Local pathway
        self.pra = PRA(embed_dim=self.feature_dim)
        self.se_local = SE_Global(in_channels=self.feature_dim)

        # Global pathway
        self.app = AdaptivePatchPooling(dim=self.feature_dim)
        self.se_global = SE_Local(in_channels=self.feature_dim)

        # Dynamic attention to fuse both pathways
        self.ddga = DDGA(dim=self.feature_dim)

        # Final SE Fusion block
        self.se_fusion = SE_Block(in_channels=self.feature_dim)

        # Final pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)[-1]  # Extract deepest feature: [B, C, H, W]
        B, C, H, W = feat.size()

        # Apply Partial Attention after feature extraction
        feat = self.partial_attention(feat)
        
        # === Local Pathway ===
        pra_feat = self.pra(feat)                           # [B, H*W, C]
        pra_feat = pra_feat.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        pra_feat = self.se_local(pra_feat)

        # === Global Pathway ===
        app_feat = self.app(feat)
        app_feat = self.se_global(app_feat)

        # === Dual Dynamic Gated Attention Fusion ===
        fused_feat = self.ddga(pra_feat, app_feat)

        # === Final Fusion Attention via SE ===
        fused_feat = self.se_fusion(fused_feat)

        # === Classification Head ===
        pooled = self.global_pool(fused_feat).view(B, C)  # [B, C]
        out = self.classifier(self.dropout(pooled))       # [B, num_classes]

        return out