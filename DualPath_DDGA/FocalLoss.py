import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight  # torch.tensor of class weights, if available

    def forward(self, input, target):
        logp = F.log_softmax(input, dim=-1)
        p = torch.exp(logp)
        loss = F.nll_loss((1 - p) ** self.gamma * logp, target, weight=self.weight)
        return loss
