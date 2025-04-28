import torch
import torch.nn as nn
import torch.nn.functional as F


# === Advanced PartialAttentionMasking ===
class AdvancedPartialAttentionMasking(nn.Module):
    def __init__(self, masking_ratio=0.5, strategy="entropy_topk"):
        super(AdvancedPartialAttentionMasking, self).__init__()
        self.masking_ratio = masking_ratio
        self.strategy = strategy

    def forward(self, x):
        B, C, H, W = x.size()

        if self.strategy == "entropy_topk":
            # 1. Compute Entropy per channel
            x_flat = x.view(B, C, -1)  # [B, C, H*W]
            probs = F.softmax(x_flat, dim=-1) + 1e-6  # Avoid log(0)
            entropy = -torch.sum(probs * probs.log(), dim=-1)  # [B, C]

            # 2. Invert entropy to get importance (low entropy → high importance)
            importance = -entropy  # [B, C]

            # 3. Select Top-K Channels
            k = int(C * (1 - self.masking_ratio))  # jumlah channel yang diloloskan
            topk_scores, topk_idx = torch.topk(importance, k, dim=1)  # [B, k]

            # 4. Build mask
            mask = torch.zeros(B, C, device=x.device)
            for b in range(B):
                mask[b, topk_idx[b]] = 1.0

            # 5. Apply mask
            mask = mask.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            x = x * mask  # hanya channel terpilih yang dipertahankan

            return x

        else:
            raise NotImplementedError(f"Strategy {self.strategy} not implemented.")
        
# === Modified PartialAttentionMasking in SemanticAwarePartialAttention ===
class SemanticAwarePartialAttention(nn.Module):
    def __init__(self, masking_ratio=0.5, topk_semantic=True):
        super(SemanticAwarePartialAttention, self).__init__()
        self.masking_ratio = masking_ratio
        self.topk_semantic = topk_semantic
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        feat = x.view(B, C, -1)  # [B, C, HW]
        feat_norm = F.normalize(feat, dim=1)  # normalize per channel

        # Calculate semantic correlation matrix
        semantic_corr = torch.bmm(feat_norm.transpose(1, 2), feat_norm)  # [B, HW, HW]

        # Aggregate semantic importance
        semantic_score = semantic_corr.mean(dim=-1)  # [B, HW]

        # Select Top-K important patches
        k = int(self.masking_ratio * H * W)

        if self.topk_semantic:
            topk_indices = torch.topk(semantic_score, k=k, dim=-1, largest=True)[1]
        else:
            topk_indices = torch.topk(semantic_score, k=k, dim=-1, largest=False)[1]

        # Create mask
        mask = torch.zeros_like(semantic_score)
        mask.scatter_(1, topk_indices, 1)
        mask = mask.unsqueeze(1)  # [B, 1, HW]

        # Apply mask
        feat = feat * mask  # [B, C, HW]
        feat = feat.view(B, C, H, W)

        return feat

class DynamicGridDSEPlus(nn.Module):
    def __init__(self, in_channels, out_channels=None, grid_sizes=[2, 4, 8]):
        super(DynamicGridDSEPlus, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels
        self.grid_sizes = grid_sizes

        self.attention_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.out_channels // 4, self.out_channels, kernel_size=1),
                nn.Sigmoid()
            ) for _ in grid_sizes
        ])

        self.final_fusion = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        attention_maps = []

        for i, grid_size in enumerate(self.grid_sizes):
            pooled = F.adaptive_avg_pool2d(x, output_size=(grid_size, grid_size))
            attn = self.attention_blocks[i](pooled)
            attn = F.interpolate(attn, size=(H, W), mode='bilinear', align_corners=False)
            attention_maps.append(attn)

        fused_attention = torch.stack(attention_maps, dim=0).mean(dim=0)  # average across grid scales
        fused_attention = self.final_fusion(fused_attention)
        fused_attention = torch.sigmoid(fused_attention)

        out = x * fused_attention
        out = out + x  # residual connection

        return out

class DynamicGridDSE(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size=2):
        super(DynamicGridDSE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        
        self.fc1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        gh, gw = self.grid_size, self.grid_size

        # Adaptive average pooling based on grid
        pooled = F.adaptive_avg_pool2d(x, output_size=(gh, gw))  # [B, C, gh, gw]
        pooled = self.fc1(pooled)
        pooled = self.relu(pooled)
        pooled = self.fc2(pooled)
        pooled = self.sigmoid(pooled)

        # Expand to original resolution
        pooled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)

        # Reweight original feature
        out = x * pooled
        return out

# === PartialAttentionMasking ===
class PartialAttentionMasking(nn.Module):
    def __init__(self, masking_ratio=0.5, strategy="topk"):
        """
        masking_ratio: berapa persen patch yang dipertahankan (0.5 berarti 50% patch aktif)
        strategy: 'topk' atau 'random'
        """
        super(PartialAttentionMasking, self).__init__()
        self.masking_ratio = masking_ratio
        self.strategy = strategy

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.size()
        feat_flat = x.view(B, C, H * W)  # [B, C, HW]

        if self.strategy == "topk":
            # Energy = Mean activation per patch
            energy = feat_flat.mean(dim=1)  # [B, HW]

            # Top-k selection
            k = int(H * W * self.masking_ratio)
            topk_vals, topk_indices = torch.topk(energy, k=k, dim=1)

            # Create binary mask
            mask = torch.zeros_like(energy)  # [B, HW]
            mask.scatter_(1, topk_indices, 1)

            # Expand mask to match feature channels
            mask = mask.unsqueeze(1)  # [B, 1, HW]
            masked_feat = feat_flat * mask  # [B, C, HW]
        
        elif self.strategy == "random":
            # Random mask
            mask = torch.rand((B, H * W), device=x.device) < self.masking_ratio
            mask = mask.float().unsqueeze(1)  # [B, 1, HW]
            masked_feat = feat_flat * mask  # [B, C, HW]
        
        else:
            raise ValueError(f"Unknown masking strategy {self.strategy}")

        # Reshape back
        masked_feat = masked_feat.view(B, C, H, W)
        return masked_feat

# === PRA untuk Local Features ===
class PRA(nn.Module):
    def __init__(self, embed_dim):
        super(PRA, self).__init__()
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, HW, C]
        x = self.norm(x)
        return x

# === PRA MultiheadAttention untuk Local Features ===
class PRA_MultiheadAttention(nn.Module):
    def __init__(self, embed_dim):
        super(PRA_MultiheadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.fc = nn.Linear(embed_dim, embed_dim)  # Fully connected to adjust output for SE
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Step 1: Flatten height and width to create seq_length
        B, C, H, W = x.size()  # B = batch_size, C = channels, H = height, W = width
        seq_len = H * W
        x = x.view(B, C, seq_len).transpose(1, 2)  # Reshape to (batch_size, seq_len, embed_dim)

        # Apply self-attention to capture positional relationships
        attention_output, _ = self.attention(x, x, x)  # Self-attention on input feature
        attention_output = self.layer_norm(attention_output + x)  # Residual connection
        output = self.fc(attention_output)  # Fully connected layer for feature adjustment
        output = self.dropout(output)  # Dropout for regularization
        return output

# ==== Dynamic Partial Regional Attention (D-PRA) ====
class D_PRA(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.size()
        seq_len = H * W
        x = x.view(B, C, seq_len).transpose(1, 2)
        attention_output, _ = self.attention(x, x, x)
        output = self.layer_norm(attention_output + x)
        output = self.fc(output)
        output = self.dropout(output)
        return output

# ====== CrossLevelSEBlock ======
class CrossLevelSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, grid_size=2):
        super(CrossLevelSEBlock, self).__init__()
        assert in_channels % 2 == 0, "in_channels harus genap karena akan di-split."

        self.local_se = DSE_Local(in_channels // 2, out_channels=in_channels // 2, grid_size=grid_size)
        self.global_se = DSE_Global(in_channels // 2, out_channels=in_channels // 2)

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
    
# === SEBlock_Enhanced ===
class SEBlock_Enhanced(nn.Module):
    def __init__(self, in_channels, out_channels=128, reduction=8):
        super(SEBlock_Enhanced, self).__init__()
        self.adjust_channels = (in_channels != out_channels)
        if self.adjust_channels:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.fc1 = nn.Conv2d(out_channels, out_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(out_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.adjust_channels:
            x = self.conv1(x)
        
        se = torch.mean(x, dim=(2, 3), keepdim=True)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)

        return x * se + x  # Residual preserve for stability
    
# === APP untuk Global Features ===
class AdaptivePatchPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.proj(self.pool(x))

# === Squeeze-Excitation Channel Attention ===
class SE_Block(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.se(x).view(b, c, 1, 1)
        return x * y.expand_as(x)

# === Local Refinement Module ===
class LocalRefinementModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.refine(x)

# === Semantic-Aware Pooling ===
class SemanticAwarePooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        weights = torch.sigmoid(self.conv(x))
        return (x * weights).sum(dim=[2, 3]) / (weights.sum(dim=[2, 3]) + 1e-6)
        #return (x * weights).sum(dim=[2, 3]) / weights.sum(dim=[2, 3])

class DSE_Global(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super(DSE_Global, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.fc1 = nn.Conv2d(out_channels, out_channels // 16, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(out_channels // 16, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        se = torch.mean(x, dim=(2, 3), keepdim=True)  # Global average pooling
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        return x * se

class DSE_Local(nn.Module):
    def __init__(self, in_channels, out_channels=128, grid_size=2):
        super(DSE_Local, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.grid_pool = nn.AdaptiveAvgPool2d(grid_size)
        self.fc1 = nn.Conv2d(out_channels, out_channels // 16, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(out_channels // 16, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        se = self.grid_pool(x)  # Pool ke grid kecil (misal 2x2)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)

        # ➡️ Tambahkan ini untuk mengatasi mismatch
        se = torch.nn.functional.interpolate(se, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return x * se

class SE_Local(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.spatial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        spatial = self.spatial_conv(x).mean(dim=[2, 3])  # agregasi spasial
        y = self.fc(torch.cat([y_avg + spatial, y_max], dim=1)).view(b, c, 1, 1)
        return x * y

class SE_Global(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
class PRA_DSE(nn.Module):
    def __init__(self, embed_dim):
        super(PRA_DSE, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.dse = DSE_Local(embed_dim)  # Dynamic SE khusus local

    def forward(self, x):
        B, C, H, W = x.size()
        seq_len = H * W
        x = x.view(B, C, seq_len).transpose(1, 2)  # [B, seq_len, C]
        
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(attn_out + x)
        x = self.fc(x)
        x = self.dropout(x)
        
        # [B, seq_len, C] → [B, C, H, W] untuk DSE
        x_reshaped = x.transpose(1, 2).view(B, C, H, W)
        x_dse = self.dse(x_reshaped)
        return x_dse

# === Feature Aware-DDGA (Dynamic Soft Fusion Dual Gating Attention) ===
class FeatureAwareDDGA(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super(FeatureAwareDDGA, self).__init__()
        self.feature_dim = feature_dim

        # MLP kecil untuk generate gate logits dari feature statistics
        self.gate_mlp = nn.Sequential(
            nn.Linear(4 * feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, pra_feat, app_feat):
        B, C, H, W = pra_feat.shape

        # === Feature Statistics Extraction ===
        pra_mean = pra_feat.mean(dim=[2, 3])  # [B, C]
        pra_var = pra_feat.var(dim=[2, 3])
        app_mean = app_feat.mean(dim=[2, 3])
        app_var = app_feat.var(dim=[2, 3])

        # Concatenate statistics
        stats = torch.cat([pra_mean, pra_var, app_mean, app_var], dim=1)  # [B, 4C]

        # Generate Gate Logits
        gate_logits = self.gate_mlp(stats)  # [B, 2]
        gate_scores = F.softmax(gate_logits, dim=1)  # [B, 2]

        # Apply Gate to Features
        gate_pra = gate_scores[:, 0].view(B, 1, 1, 1)
        gate_app = gate_scores[:, 1].view(B, 1, 1, 1)

        fused_feat = gate_pra * pra_feat + gate_app * app_feat

        # === Optional: Add residual fusion
        fused_feat = fused_feat + 0.5 * (pra_feat + app_feat)

        # === Optional: Calculate entropy loss for gate diversity
        entropy_loss = -(gate_scores * gate_scores.log()).sum(dim=1).mean()

        return fused_feat, entropy_loss
    
# Dual Dynamic Gated Attention Fusion Entropy Regularization Calculation
class DDGA_Entropy(nn.Module):
    def __init__(self, dim):
        super(DDGA_Entropy, self).__init__()
        self.gate_local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )
        self.gate_global = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )
        self.fusion = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, local_feat, global_feat):
        # Gated local and global feature maps
        l_gate = self.gate_local(local_feat) * local_feat
        g_gate = self.gate_global(global_feat) * global_feat

        # 🔥 Adjust size if needed (upsampling global_feat)
        if l_gate.size()[2:] != g_gate.size()[2:]:
            g_gate = F.interpolate(g_gate, size=(l_gate.size(2), l_gate.size(3)), mode='bilinear', align_corners=False)

        # Fuse gated features
        fused = torch.cat([l_gate, g_gate], dim=1)
        fused = self.fusion(fused)

        # 🔥 Entropy Regularization Calculation
        # Take the gating maps before element-wise multiplication
        l_score = self.gate_local(local_feat)
        g_score = self.gate_global(global_feat)

        # Normalize scores to probability distribution per location
        total_score = l_score + g_score + 1e-8  # Avoid division by zero
        l_prob = l_score / total_score
        g_prob = g_score / total_score

        # Entropy per pixel
        entropy_map = -(l_prob * torch.log(l_prob + 1e-8) + g_prob * torch.log(g_prob + 1e-8))
        entropy_loss = entropy_map.mean()

        return fused, entropy_loss

# === DSF-DDGA (Dynamic Soft Fusion Dual Gating Attention) ===
class DDGA_DSF(nn.Module):
    def __init__(self, feature_dim, use_attention=False):
        super(DDGA_DSF, self).__init__()
        self.feature_dim = feature_dim
        self.use_attention = use_attention

        # Gating network: predict soft routing weights
        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim * 2, 2)  # output 2 routing scores: PRA and APP
        )

        # Optional attention layer untuk fusi feature
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
            )
        else:
            self.attention = None

    def forward(self, pra_feat, app_feat):
        # 🔥 Penyesuaian ukuran feature map sebelum concatenation
        if pra_feat.size()[2:] != app_feat.size()[2:]:
            app_feat = F.interpolate(app_feat, size=pra_feat.shape[2:], mode='bilinear', align_corners=False)

        # Step 1: Concatenate features
        fusion = torch.cat([pra_feat, app_feat], dim=1)  # [B, 2C, H, W]

        # Step 2: Predict gating scores
        gate_logits = self.gate_fc(fusion)
        gate_scores = F.softmax(gate_logits, dim=1)  # [B, 2]

        gate_pra = gate_scores[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [B, 1, 1, 1]
        gate_app = gate_scores[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [B, 1, 1, 1]

        # Step 3: Weighted sum
        fused_feat = gate_pra * pra_feat + gate_app * app_feat

        # Step 4: Residual shortcut (optional, untuk stabilisasi)
        fused_feat = fused_feat + 0.5 * (pra_feat + app_feat)

        # Step 5: Optional lightweight attention fusion
        if self.attention is not None:
            fused_feat = self.attention(torch.cat([pra_feat, app_feat], dim=1))

        # Step 6: Compute entropy regularization
        entropy = -(gate_scores * gate_scores.log()).sum(dim=1).mean()

        return fused_feat, entropy, gate_scores
    
# === Dual Dynamic Gated Attention Fusion ===
class DDGA(nn.Module):
    def __init__(self, dim):
        super(DDGA, self).__init__()
        self.gate_local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()  # Mengontrol kontribusi local
        )
        self.gate_global = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()  # Mengontrol kontribusi global
        )
        self.fusion = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, local_feat, global_feat):
        l_gate = self.gate_local(local_feat) * local_feat
        g_gate = self.gate_global(global_feat) * global_feat

        # 🔥 Tambahkan upsampling global_feat
        if l_gate.size()[2:] != g_gate.size()[2:]:  # Kalau ukuran spatial beda
            g_gate = F.interpolate(g_gate, size=(l_gate.size(2), l_gate.size(3)), mode='bilinear', align_corners=False)

        fused = torch.cat([l_gate, g_gate], dim=1)
        return self.fusion(fused)

# ==== Confidence Aware Fusion Block ====
class ConfidenceAwareFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.local_fc = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.AdaptiveAvgPool2d(1)

    def forward(self, local_feat, global_feat):
        local_conf = self.local_fc(local_feat).view(local_feat.size(0), -1).mean(dim=1, keepdim=True)
        global_conf = self.global_fc(global_feat).view(global_feat.size(0), -1).mean(dim=1, keepdim=True)
        sum_conf = local_conf + global_conf + 1e-6
        alpha = local_conf / sum_conf
        beta = global_conf / sum_conf

        fused = alpha.view(-1, 1, 1, 1) * local_feat + beta.view(-1, 1, 1, 1) * global_feat
        
        return fused, alpha, beta
    
# === Cross Dual Dynamic Gated Attention Fusion ===
class CrossLevelDualGatedAttention(nn.Module):
    def __init__(self, dim):
        super(CrossLevelDualGatedAttention, self).__init__()
        self.local_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.global_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.cross_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )
        # INI YANG DIRUBAH:
        self.fusion_proj = nn.Conv2d(dim * 2, dim * 2, kernel_size=1)  # channel tetap 2C

    def forward(self, local_feat, global_feat):
        B, C, H, W = local_feat.shape

        local_proj = self.local_proj(local_feat)
        global_proj = self.global_proj(global_feat)

        if global_proj.shape[-2:] != (H, W):
            global_proj = torch.nn.functional.interpolate(global_proj, size=(H, W), mode='bilinear', align_corners=False)

        concat_feat = torch.cat([local_proj, global_proj], dim=1)
        gate = self.cross_gate(concat_feat)

        gated_local = local_proj * gate
        gated_global = global_proj * (1 - gate)

        fused_feat = torch.cat([gated_local, gated_global], dim=1)  # [B, 2C, H, W]
        out = self.fusion_proj(fused_feat)

        return out

#Modifikasi APP (HierarchicalAPP) menyusun informasi dari multi-skala patch:
class HierarchicalAPP(nn.Module):
    def __init__(self, dim):
        super(HierarchicalAPP, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)  # Global
        self.pool2 = nn.AdaptiveAvgPool2d(2)  # Semi-global
        self.pool3 = nn.AdaptiveAvgPool2d(4)  # Regional

        self.conv = nn.Sequential(
            nn.Conv2d(dim * (1 + 4 + 16), dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        feat1 = self.pool1(x)                         # [B, C, 1, 1]
        feat2 = self.pool2(x).view(b, c, -1, 1)        # [B, C, 4, 1]
        feat3 = self.pool3(x).view(b, c, -1, 1)        # [B, C, 16, 1]
        concat = torch.cat([feat1.view(b, c, 1, 1), feat2, feat3], dim=2)
        concat = concat.view(b, -1, 1, 1)              # Merge channel
        return self.conv(concat)

# Modifikasi APP Menambahkan encoding posisi relatif agar APP tidak buta arah.
class PositionalAPP(nn.Module):
    def __init__(self, dim, grid_size=(7, 7)):
        super(PositionalAPP, self).__init__()
        self.grid_size = grid_size
        self.pos_embed = nn.Parameter(torch.randn(1, dim, *grid_size))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = x + F.interpolate(self.pos_embed, size=x.shape[2:], mode='bilinear')
        return self.conv(self.pool(x))

# Modifikasi APP Modul pooling dipandu oleh self-attention ringan 
# (mirip CBAM channel attention) untuk memilih fitur dominan sebelum pooling.
class AGAPP(nn.Module):
    def __init__(self, dim):
        super(AGAPP, self).__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        att_map = self.att(x)
        x = x * att_map
        return self.pool(x)

class AdaptiveContextAPP(nn.Module):
    def __init__(self, dim):
        super(AdaptiveContextAPP, self).__init__()
        self.local_att = nn.Conv2d(dim, 1, kernel_size=3, padding=1)  # Saliency estimation
        self.sigmoid = nn.Sigmoid()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))  # Tidak langsung 1x1

    def forward(self, x):
        saliency = self.sigmoid(self.local_att(x))
        x = x * saliency  # Local modulation
        pooled = self.adaptive_pool(x)  # 2x2 feature map
        return pooled
    
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