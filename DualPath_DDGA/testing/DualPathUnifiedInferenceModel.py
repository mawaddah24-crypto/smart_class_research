import torch
import torch.nn as nn
import torch.nn.functional as F
from WeightedPartialAttention import WeightedPartialAttention
from dummy_gaze_pose_generator import generate_dummy_gaze_pose_importance

class DualPathUnifiedInferenceModel(nn.Module):
    def __init__(self, base_model, gaze_model=None, pose_model=None, masking_ratio=0.5, alpha=0.6, beta=0.2, gamma=0.2):
        """
        base_model: trained base DualPath model (e.g., DualPath_PartialAttentionModif)
        gaze_model: pretrained gaze estimation model (or None for dummy)
        pose_model: pretrained pose estimation model (or None for dummy)
        """
        super(DualPathUnifiedInferenceModel, self).__init__()
        
        # Backbone & Pathways from the base model
        self.backbone = base_model.backbone
        self.pra = base_model.pra
        self.app = base_model.app
        self.ddga = base_model.ddga
        self.se_fusion = base_model.se_fusion
        self.global_pool = base_model.global_pool
        self.classifier = base_model.classifier
        self.dropout = base_model.dropout
        
        # Attention mechanism
        self.weighted_pa = WeightedPartialAttention(masking_ratio, alpha, beta, gamma)
        
        # Gaze and Pose Models (optional)
        self.gaze_model = gaze_model
        self.pose_model = pose_model

    def forward(self, x):
        """
        x: input tensor (B, 3, 224, 224)
        """
        B = x.shape[0]
        device = x.device

        # Feature extraction
        feat = self.backbone(x)[-1]  # Shape: [B, C, 7, 7]

        # Generate importance maps
        if self.gaze_model and self.pose_model:
            gaze_importance, pose_importance = self.generate_gaze_pose_importance_real(x)
        else:
            gaze_importance, pose_importance = generate_dummy_gaze_pose_importance(B, feat.size(2), feat.size(3), device=device)

        # Apply Weighted Partial Attention
        feat = self.weighted_pa(feat, gaze_importance, pose_importance)

        # Local Pathway
        pra_feat = self.pra(feat).transpose(1, 2).view(feat.shape)
        pra_feat = self.se_fusion(pra_feat)

        # Global Pathway
        app_feat = self.app(feat)
        app_feat = self.se_fusion(app_feat)

        # Fusion
        fused_feat = self.ddga(pra_feat, app_feat)
        fused_feat = self.se_fusion(fused_feat)

        # Classification
        pooled = self.global_pool(fused_feat).view(B, -1)
        out = self.classifier(self.dropout(pooled))
        
        return out

    def generate_gaze_pose_importance_real(self, x):
        """
        If real gaze and pose model available.
        Implement actual gaze/pose inference here.
        For now, placeholder using dummy.
        """
        # Implement if real model is ready
        # Ex: self.gaze_model(x), self.pose_model(x)
        B = x.size(0)
        device = x.device
        return generate_dummy_gaze_pose_importance(B, 7, 7, device)
