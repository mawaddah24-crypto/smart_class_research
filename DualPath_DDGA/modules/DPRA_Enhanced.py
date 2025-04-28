import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, embed_dim]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class AutoAdaptiveSELocal(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.reduction = reduction

    def forward(self, x):
        b, c, h, w = x.size()
        grid_size = 2 if min(h, w) <= 7 else 4

        avg_pool = nn.AdaptiveAvgPool2d(grid_size)(x)
        fc1 = nn.Conv2d(c, c // self.reduction, kernel_size=1)
        relu = nn.ReLU(inplace=True)
        fc2 = nn.Conv2d(c // self.reduction, c, kernel_size=1)

        se = fc1(avg_pool)
        se = relu(se)
        se = fc2(se)
        se = torch.sigmoid(se)
        se = F.interpolate(se, size=(h, w), mode='bilinear', align_corners=False)

        return x * se

class DPRA_Enhanced(nn.Module):
    def __init__(self, embed_dim, masking_ratio=0.5, use_masking=True):
        super().__init__()
        self.masking_ratio = masking_ratio
        self.use_masking = use_masking

        self.positional_encoding = PositionalEncoding(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.se_local = AutoAdaptiveSELocal(embed_dim)

    def enable_masking(self):
        self.use_masking = True

    def disable_masking(self):
        self.use_masking = False

    def train(self, mode: bool = True):
        """Override .train() untuk auto switch masking."""
        super().train(mode)
        if mode:
            self.enable_masking()
        else:
            self.disable_masking()

    def forward(self, x):
        B, C, H, W = x.size()
        seq_len = H * W

        x = x.view(B, C, seq_len).transpose(1, 2)  # [B, Seq, C]
        x = self.positional_encoding(x)

        if self.use_masking:
            energy = x.mean(dim=-1)  # [B, Seq]
            k = max(1, int(seq_len * self.masking_ratio))
            topk_indices = torch.topk(energy, k=k, dim=1).indices

            mask = torch.zeros_like(energy)
            mask.scatter_(1, topk_indices, 1)
            mask = mask.unsqueeze(-1)
            x = x * mask

        attn_output, _ = self.attention(x, x, x)
        output = self.layer_norm(attn_output + x)
        output = self.fc(output)
        output = self.dropout(output)

        output = output.transpose(1, 2).view(B, C, H, W)
        output = self.se_local(output)

        return output


#self.pra = DPRA_Enhanced(embed_dim=self.feature_dim, masking_ratio=0.5, use_masking=True)
#self.pra = DPRA_Enhanced(embed_dim=self.feature_dim, use_masking=False)
