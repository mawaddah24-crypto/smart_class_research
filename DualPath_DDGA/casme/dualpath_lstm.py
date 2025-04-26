import torch
import torch.nn as nn
from DualPathModel import DualPath_Baseline  # Atau model modif Anda

class DualPathLSTM(nn.Module):
    def __init__(self, num_classes=5, feature_dim=256, lstm_hidden=128, lstm_layers=1):
        super(DualPathLSTM, self).__init__()
        self.backbone = DualPath_Baseline(num_classes=num_classes, pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove classifier head
        self.feature_dim = feature_dim

        self.lstm = nn.LSTM(input_size=self.feature_dim,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=False)

        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len, C, H, W]
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)

        features = self.backbone(x)  # [batch_size*seq_len, feature_dim]
        features = features.view(batch_size, seq_len, -1)  # [batch_size, seq_len, feature_dim]

        lstm_out, _ = self.lstm(features)  # [batch_size, seq_len, hidden]
        final_out = lstm_out[:, -1, :]  # Take last frame output
        out = self.classifier(final_out)
        return out
