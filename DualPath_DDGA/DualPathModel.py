import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from module import (
    PartialAttentionMasking, 
    DSE_Global, 
    DSE_Local, 
    DDGA, DDGA_Entropy, DDGA_DSF, 
    AGAPP, 
    SEBlock_Enhanced, 
    PRA_MultiheadAttention,
    PRA, SE_Global, SE_Local, SE_Block, AdaptivePatchPooling, SemanticAwarePooling)
                    
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
    
class DualPath(nn.Module):
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', pretrained=True, 
                 in_channels=3, delay_ddga=True):
        super(DualPath, self).__init__()
        
        self.delay_ddga = delay_ddga  # True: delay DDGA activation
        # Backbone: EfficientViT pretrained
        self.backbone = create_model(backbone_name, pretrained=pretrained, features_only=True, in_chans=in_channels)
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']  # usually 128

        # Local pathway
        self.pra = PRA_MultiheadAttention(embed_dim=self.feature_dim)

        # Global pathway
        self.app = AGAPP(dim=self.feature_dim)
       
        # Final SE Fusion block
        self.se_fusion = CrossLevelSEBlock(in_channels=self.feature_dim * 2)
        
        # Reduction after fusion (optional)
        self.reduction_conv = nn.Conv2d(self.feature_dim * 2, self.feature_dim, kernel_size=1)
        # Final pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)[-1]  # Extract deepest feature: [B, C, H, W]
        B, C, H, W = feat.size()

        # === Local Pathway ===
        pra_feat = self.pra(feat).transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)

        # === Global Pathway ===
        app_feat = self.app(feat)  # Mungkin keluar (B, C, 1, 1)

        # === Fix dimension mismatch ===
        if app_feat.shape[-2:] != pra_feat.shape[-2:]:
            app_feat = F.interpolate(app_feat, size=pra_feat.shape[2:], mode='bilinear', align_corners=False)

        # === Fusion ===
        fused_feat = torch.cat([pra_feat, app_feat], dim=1)  # Sekarang aman concat
        fused_feat = self.se_fusion(fused_feat)
        fused_feat = self.reduction_conv(fused_feat)
        # === Classification Head ===
        pooled = self.global_pool(fused_feat).flatten(1)
        out = self.classifier(self.dropout(pooled))       # [B, num_classes]

        return out

class DualPath_DRM(nn.Module):
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', pretrained=True, 
                 in_channels=3, use_attention=True):
        super(DualPath_DRM, self).__init__()

        # Backbone
        self.backbone = create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels
        )
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']  # Example: 128

        # Local and Global pathways
        self.partial_attention = PartialAttentionMasking(masking_ratio=0.5, strategy="topk")
        self.pra = PRA_MultiheadAttention(embed_dim=self.feature_dim)
        self.se_local = DSE_Local(in_channels=self.feature_dim, out_channels=self.feature_dim)
        self.app = AGAPP(dim=self.feature_dim)
        self.se_global = DSE_Global(in_channels=self.feature_dim, out_channels=self.feature_dim)

        # Fusion
        self.ddga = DDGA_DSF(feature_dim=self.feature_dim, use_attention=use_attention)
        
        # FIX INI:
        fusion_channels = self.feature_dim * 2  # 128 * 4 = 512
        self.reduction_conv = nn.Conv2d(fusion_channels, self.feature_dim, kernel_size=1)

        # ✨ Dynamic Recalibration Module
        self.drm = DynamicRecalibration(channels=self.feature_dim)

        # Pooling and Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)[-1]
        B, C, H, W = feat.size()

        feat = self.partial_attention(feat)

        # Local pathway
        pra_feat = self.pra(feat).transpose(1, 2).view(B, C, H, W)
        pra_feat = self.se_local(pra_feat)

        # Global pathway
        app_feat = self.app(feat)
        app_feat = self.se_global(app_feat)
        if app_feat.shape[-2:] != pra_feat.shape[-2:]:
            app_feat = F.interpolate(app_feat, size=pra_feat.shape[2:], mode='bilinear', align_corners=False)

        # Fusion with DDGA
        fused_feat, entropy_loss = self.ddga(pra_feat, app_feat)
        #print(f"[DEBUG] Fused Feature Shape after DDGA: {fused_feat.shape}")
        # Reduction + Dynamic Recalibration
        fused_feat = self.reduction_conv(fused_feat)
        fused_feat = self.drm(fused_feat)

        # Pooling and classification
        pooled = self.global_pool(fused_feat).flatten(1)
        out = self.classifier(self.dropout(pooled))

        return {
            'logits': out,
            'entropy_loss': entropy_loss
        }
        
class DualPath_DDGA(nn.Module):
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', pretrained=True, in_channels=3):
        super(DualPath_DDGA, self).__init__()
        
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
        self.ddga = DDGA_DSF(feature_dim=self.feature_dim,use_attention=True)  
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
        #fused_feat = self.ddga(pra_feat, app_feat)
        fused_feat, entropy_loss = self.ddga(pra_feat, app_feat)
        
        # === Final Fusion Attention via SE ===
        fused_feat = self.se_fusion(fused_feat)

        # === Classification Head ===
        pooled = self.global_pool(fused_feat).view(B, C)  # [B, C]
        out = self.classifier(self.dropout(pooled))       # [B, num_classes]

        return {
            'logits': out,
            'entropy_loss': entropy_loss
            }

class DualPath_Fusion(nn.Module):
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', 
                 pretrained=True, in_channels=3, use_attention=True):
        super(DualPath_Fusion, self).__init__()
        
        
        # Backbone: EfficientViT pretrained
        self.backbone = create_model(backbone_name, pretrained=pretrained, features_only=True, 
                                     in_chans=in_channels)
        
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']  # usually 128
        self.partial_attention = PartialAttentionMasking(masking_ratio=0.5, strategy="topk")
        # Local pathway
        self.pra = PRA_MultiheadAttention(embed_dim=self.feature_dim)
        self.se_local = DSE_Local(in_channels=self.feature_dim, out_channels=self.feature_dim)

        # Global pathway
        self.app = AGAPP(dim=self.feature_dim)
        self.se_global = DSE_Global(in_channels=self.feature_dim, out_channels=self.feature_dim)

        # Dynamic attention to fuse both pathways
        self.ddga = DDGA_DSF(feature_dim=self.feature_dim,use_attention=True)  
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
        fused_feat, entropy_loss = self.ddga(pra_feat, app_feat)
        
        # === Final Fusion Attention via SE ===
        fused_feat = self.se_fusion(fused_feat)

        # === Classification Head ===
        pooled = self.global_pool(fused_feat).view(B, C)  # [B, C]
        out = self.classifier(self.dropout(pooled))       # [B, num_classes]

        return {
            'logits': out,
            'entropy_loss': entropy_loss
            }
    
class DualPath_Baseline(nn.Module):
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', pretrained=True, 
                 in_channels=3,delay_ddga=True):
        super(DualPath_Baseline, self).__init__()
        
        self.delay_ddga = delay_ddga  # True: delay DDGA activation
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
        if self.delay_ddga:
            # Fusion modules
            self.simple_fusion = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1)
        else:
            self.ddga = DDGA(dim=self.feature_dim)
            #self.ddga = DDGA_Entropy(dim=self.feature_dim)

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
        if self.delay_ddga:
            # Stage 1: simple fusion
            fused_feat = pra_feat + app_feat
            fused_feat = self.simple_fusion(fused_feat)
            entropy = None
        else:
            # Stage 2: adaptive fusion with DDGA
            fused_feat = self.ddga(pra_feat, app_feat)
            
        # === Final Fusion Attention via SE ===
        fused_feat = self.se_fusion(fused_feat)

        # === Classification Head ===
        pooled = self.global_pool(fused_feat).view(B, C)  # [B, C]
        out = self.classifier(self.dropout(pooled))       # [B, num_classes]

        return out

class DualPath_Baseline_DSE(nn.Module):
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', pretrained=True, 
                 in_channels=3, delay_ddga=True):
        super(DualPath_Baseline_DSE, self).__init__()
        
        self.delay_ddga = delay_ddga  # True: delay DDGA activation
        # Backbone: EfficientViT pretrained
        self.backbone = create_model(backbone_name, pretrained=pretrained, features_only=True, in_chans=in_channels)
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']  # usually 128

        # Local pathway
        self.pra = PRA(embed_dim=self.feature_dim)
        self.se_local = DSE_Local(in_channels=self.feature_dim, out_channels=self.feature_dim)
       
        # Global pathway
        self.app = AdaptivePatchPooling(dim=self.feature_dim)
        self.se_global = DSE_Global(in_channels=self.feature_dim, out_channels=self.feature_dim)

        # Dynamic attention to fuse both pathways
        if self.delay_ddga:
            # Fusion modules
            self.simple_fusion = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1)
        else:
            self.ddga = DDGA(dim=self.feature_dim)
            #self.ddga = DDGA_Entropy(dim=self.feature_dim)
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
        if self.delay_ddga:
            # Stage 1: simple fusion
            fused_feat = pra_feat + app_feat
            fused_feat = self.simple_fusion(fused_feat)
            entropy = None
        else:
            # Stage 2: adaptive fusion with DDGA
            fused_feat = self.ddga(pra_feat, app_feat)

        # === Final Fusion Attention via SE ===
        fused_feat = self.se_fusion(fused_feat)

        # === Classification Head ===
        pooled = self.global_pool(fused_feat).view(B, C)  # [B, C]
        out = self.classifier(self.dropout(pooled))       # [B, num_classes]

        return out
    
class DualPath_PartialAttentionModif(nn.Module):
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', pretrained=True, 
                 in_channels=3, delay_ddga=True):
        super(DualPath_PartialAttentionModif, self).__init__()
        
        self.delay_ddga = delay_ddga  # True: delay DDGA activation
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
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', pretrained=True, 
                 in_channels=3,delay_ddga=True):
        super(DualPath_Base_Partial, self).__init__()
        
        self.delay_ddga = delay_ddga  # True: delay DDGA activation
        # Backbone: EfficientViT pretrained
        self.backbone = create_model(backbone_name, pretrained=pretrained, features_only=True, in_chans=in_channels)
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']  # usually 128

        self.partial_attention = PartialAttentionMasking(masking_ratio=0.5, strategy="topk")
        
        # Local pathway
        self.pra = PRA(embed_dim=self.feature_dim)
        self.se_local = SE_Global(in_channels=self.feature_dim)

        # Global pathway
        self.app = AdaptivePatchPooling(dim=self.feature_dim)
        self.se_global = SE_Local(in_channels=self.feature_dim)

        # Dynamic attention to fuse both pathways
        if self.delay_ddga:
            # Fusion modules
            self.simple_fusion = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1)
        else:
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
        if self.delay_ddga:
            # Stage 1: simple fusion
            fused_feat = pra_feat + app_feat
            fused_feat = self.simple_fusion(fused_feat)
            entropy = None
        else:
            # Stage 2: adaptive fusion with DDGA
            fused_feat = self.ddga(pra_feat, app_feat)

        # === Final Fusion Attention via SE ===
        fused_feat = self.se_fusion(fused_feat)

        # === Classification Head ===
        pooled = self.global_pool(fused_feat).view(B, C)  # [B, C]
        out = self.classifier(self.dropout(pooled))       # [B, num_classes]

        return out

class DualPath_PartialAttentionSAP(nn.Module):
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', pretrained=True, 
                 in_channels=3,delay_ddga=True):
        super(DualPath_PartialAttentionSAP, self).__init__()

        self.delay_ddga = delay_ddga  # True: delay DDGA activation
        # Backbone: EfficientViT pretrained
        self.backbone = create_model(backbone_name, pretrained=pretrained, features_only=True, in_chans=in_channels)
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']  # usually 128


        # Partial Attention Masking
        self.partial_attention = PartialAttentionMasking(masking_ratio=0.5, strategy="topk")  # atau SAPA jika sudah

        # Local and Global Pathways
        self.pra = PRA(embed_dim=self.feature_dim)
        self.app = AGAPP(dim=self.feature_dim)

        # Dynamic Spatial Excitation (Local & Global)
        self.dse_local = DSE_Local(in_channels=self.feature_dim, out_channels=self.feature_dim)
        self.dse_global = DSE_Global(in_channels=self.feature_dim, out_channels=self.feature_dim)

        # Dual Dynamic Gated Attention
        self.ddga = DDGA(dim=self.feature_dim)

        # Semantic-Aware Pooling (NEW)
        self.semantic_pool = SemanticAwarePooling(in_channels=self.feature_dim)

        # Dropout and Classifier
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # Backbone
        feat = self.backbone(x)[-1]  # Feature map [B, C, H, W]

        # Partial Attention
        feat = self.partial_attention(feat)

        # Local Path
        pra_feat = self.pra(feat).transpose(1, 2).view(feat.shape)
        pra_feat = self.dse_local(pra_feat)

        # Global Path
        app_feat = self.app(feat)
        app_feat = self.dse_global(app_feat)

        # Fusion
        fused_feat = self.ddga(pra_feat, app_feat)

        # Semantic-Aware Pooling
        pooled = self.semantic_pool(fused_feat)

        # Classification
        out = self.classifier(self.dropout(pooled))

        return out