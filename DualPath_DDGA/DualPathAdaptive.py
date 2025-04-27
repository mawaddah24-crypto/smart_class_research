import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
import os, csv
from module import (D_PRA, 
                    AdaptiveContextAPP, 
                    DSE_Local, 
                    ConfidenceAwareFusion,
                    DynamicRecalibration)


# === Unified Partial Attention Masking ===
class PartialAttentionMasking(nn.Module):
    def __init__(self, masking_ratio=0.5, strategy="spatial_topk"):
        super(PartialAttentionMasking, self).__init__()
        self.masking_ratio = masking_ratio
        self.strategy = strategy

    def forward(self, x):
        B, C, H, W = x.size()

        if self.strategy == "spatial_topk":
            feat_flat = x.view(B, C, H * W)
            energy = feat_flat.mean(dim=1)  # [B, HW]
            k = int(H * W * self.masking_ratio)
            topk_vals, topk_indices = torch.topk(energy, k=k, dim=1)
            mask = torch.zeros_like(energy)
            mask.scatter_(1, topk_indices, 1)
            mask = mask.unsqueeze(1)
            masked_feat = feat_flat * mask
            masked_feat = masked_feat.view(B, C, H, W)
            return masked_feat

        elif self.strategy == "spatial_random":
            feat_flat = x.view(B, C, H * W)
            mask = torch.rand((B, H * W), device=x.device) < self.masking_ratio
            mask = mask.float().unsqueeze(1)
            masked_feat = feat_flat * mask
            masked_feat = masked_feat.view(B, C, H, W)
            return masked_feat

        elif self.strategy == "channel_entropy":
            x_flat = x.view(B, C, -1)
            probs = F.softmax(x_flat, dim=-1) + 1e-6
            entropy = -torch.sum(probs * probs.log(), dim=-1)
            importance = -entropy
            k = int(C * (1 - self.masking_ratio))
            topk_scores, topk_idx = torch.topk(importance, k, dim=1)
            mask = torch.zeros(B, C, device=x.device)
            for b in range(B):
                mask[b, topk_idx[b]] = 1.0
            mask = mask.unsqueeze(-1).unsqueeze(-1)
            x = x * mask
            return x

        else:
            raise ValueError(f"Unknown masking strategy {self.strategy}")

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


# ==== Final DualPathModel for Ablation ====
class DualPathAdaptiveAblation(nn.Module):
    def __init__(self, num_classes=7, backbone_name='efficientvit_b1.r224_in1k', 
                 pretrained=True, in_channels=3,
                 use_pam=True, use_dpra=True, use_dse_local=True,
                 use_adaptive_app=True, use_confidence_fusion=True, use_drm=True):
        super().__init__()

        self.use_pam = use_pam
        self.use_dpra = use_dpra
        self.use_dse_local = use_dse_local
        self.use_adaptive_app = use_adaptive_app
        self.use_confidence_fusion = use_confidence_fusion
        self.use_drm = use_drm

        self.backbone = create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels
        )
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']  # ex: 128
        
        if use_pam:
            self.pam = PartialAttentionMasking(masking_ratio=0.5, strategy="channel_entropy")

        if use_dpra:
            self.pra = D_PRA(embed_dim=self.feature_dim)
        else:
            self.simple_fc = nn.Linear(self.feature_dim, self.feature_dim)

        if use_dse_local:
            self.dse_local = DSE_Local(in_channels=self.feature_dim, out_channels=self.feature_dim)

        if use_adaptive_app:
            self.app = AdaptiveContextAPP(dim=self.feature_dim)
        else:
            self.app = SimpleGlobalPooling(dim=self.feature_dim)

        if use_confidence_fusion:
            self.conf_fusion = ConfidenceAwareFusion(feature_dim=self.feature_dim)
        else:
            self.simple_fusion = SimpleSumFusion()

        if use_drm:
            self.drm = DynamicRecalibration(channels=self.feature_dim)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)[-1]  # [B, C, H, W]
        B, C, H, W = feat.size()
        
        if self.use_pam:
            feat = self.pam(feat)

        if self.use_dpra:
            pra_feat = self.pra(feat)
            B, seq_len, C = pra_feat.size()
            H = W = int(seq_len ** 0.5)
            pra_feat = pra_feat.transpose(1, 2).view(B, C, H, W)
        else:
            pra_feat = self.simple_fc(feat.flatten(2).transpose(1, 2))
            B, seq_len, C = pra_feat.size()
            H = W = int(seq_len ** 0.5)
            pra_feat = pra_feat.transpose(1, 2).view(B, C, H, W)

        if self.use_dse_local:
            pra_feat = self.dse_local(pra_feat)

        app_feat = self.app(feat)

        if self.use_confidence_fusion:
            fused_feat = self.conf_fusion(pra_feat, app_feat)
        else:
            fused_feat = self.simple_fusion(pra_feat, app_feat)

        if self.use_drm:
            fused_feat = self.drm(fused_feat)

        pooled = self.global_pool(fused_feat).flatten(1)
        out = self.classifier(self.dropout(pooled))

        return out
                        
# ==== Final DualPathModel ====
class DualPathAdaptive(nn.Module):
    def __init__(self, num_classes=7,
                 backbone_name='efficientvit_b1.r224_in1k', pretrained=True, 
                 in_channels=3, use_drm=True, verbose=False, log_csv_path=None):
        super().__init__()
        
        self.use_drm = use_drm
        self.verbose = verbose
        self.log_csv_path = log_csv_path
        self.step_counter = 0
        
        self.backbone = create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels
        )
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']  # ex: 128
        
        self.pam = PartialAttentionMasking(masking_ratio=0.5, strategy="channel_entropy")
        self.pra = D_PRA(embed_dim=self.feature_dim)
        self.dse_local = DSE_Local(in_channels=self.feature_dim, out_channels=self.feature_dim)
        self.app = AdaptiveContextAPP(dim=self.feature_dim)
        self.conf_fusion = ConfidenceAwareFusion(feature_dim=self.feature_dim)
        self.use_drm = use_drm
        if use_drm:
            self.drm = DynamicRecalibration(channels=self.feature_dim)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        if self.log_csv_path:
            # Buat foldernya kalau belum ada
            folder = os.path.dirname(log_csv_path)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)
            
            # Inisialisasi CSV file header
            if not os.path.exists(log_csv_path):
                with open(log_csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'Step', 'PAM_Mean', 'PRA_Mean', 'APP_Mean', 
                        'Fusion_Mean', 'DRM_Mean', 'Fusion_Alpha_PRA', 'Fusion_Beta_APP'
                    ])

    def log_to_csv(self, pam_mean, pra_mean, app_mean, fusion_mean, drm_mean, alpha, beta):
            if self.log_csv_path:
                with open(self.log_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.step_counter, pam_mean, pra_mean, app_mean,
                        fusion_mean, drm_mean, alpha, beta
                    ])
                self.step_counter += 1
            
    def forward(self, x):
        feat = self.backbone(x)[-1]  # [B, C, H, W]
        B, C, H, W = feat.size()
        
        feat = self.pam(feat)
        pam_mean = feat.mean().item()

        # Local Branch
        pra_feat = self.pra(feat)
        B, seq_len, C = pra_feat.size()
        H = W = int(seq_len ** 0.5)
        pra_feat = pra_feat.transpose(1, 2).view(B, C, H, W)
        pra_feat = self.dse_local(pra_feat)
        pra_mean = pra_feat.mean().item()

        # Global Branch
        app_feat = self.app(feat)
        app_mean = app_feat.mean().item()

        # Fusion
        fused_feat, alpha, beta = self.conf_fusion(pra_feat, app_feat)
        fusion_mean = fused_feat.mean().item()

        # DRM
        if self.use_drm:
            fused_feat = self.drm(fused_feat)
        drm_mean = fused_feat.mean().item() if self.use_drm else 0
        
        if self.verbose:
            self.log_feature("PAM Output", feat)
            self.log_feature("D-PRA + DSE_Local Output", pra_feat)
            self.log_feature("AdaptiveContextAPP Output", app_feat)
            self.log_feature("Fusion Output", fusion_mean)
            self.log_feature("DRM", drm_mean)
            
        # Log to CSV
        self.log_to_csv(
            pam_mean, pra_mean, app_mean, fusion_mean, drm_mean,
            alpha.mean().item(), beta.mean().item()
        )

        pooled = self.global_pool(fused_feat).flatten(1)
        out = self.classifier(self.dropout(pooled))

        return out

# ==== Dummy Data Test ====
if __name__ == "__main__":
    # Dummy Input
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch size 4, RGB image, 224x224
    # Model
    model = DualPathAdaptive(num_classes=7, feature_dim=128, use_drm=True, verbose=True)
    output = model(dummy_input)
    print(model(dummy_input).shape)
