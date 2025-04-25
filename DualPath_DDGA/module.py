import torch
import torch.nn as nn
import torch.nn.functional as F

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

# === SEBlock_Enhanced ===
class SEBlock_Enhanced(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super(SEBlock_Enhanced, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Add a conv layer to increase channels if needed
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.fc1 = nn.Conv2d(out_channels, out_channels // 16, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(out_channels // 16, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply convolution to adjust channels if necessary
        x = self.conv1(x)  # Adjust channels to `out_channels`
        
        # Global average pooling
        se = torch.mean(x, dim=(2, 3), keepdim=True)  # Global average pooling
        
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)

        return x * se  # Scale input by SE weights
    
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
        return (x * weights).sum(dim=[2, 3]) / weights.sum(dim=[2, 3])

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
    