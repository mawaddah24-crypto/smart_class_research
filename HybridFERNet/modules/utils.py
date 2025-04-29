# utils/metrics.py
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR, ReduceLROnPlateau

# Hitung akurasi top-1
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Confusion Matrix (optional untuk evaluasi lebih lanjut)
def confusion_matrix(preds, labels, num_classes):
    cm = torch.zeros(num_classes, num_classes)
    for p, t in zip(preds, labels):
        cm[t, p] += 1
    return cm

# utils/schedulers
# Pilih scheduler
def get_scheduler(optimizer, config):
    if config['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'step':
        scheduler = StepLR(optimizer, step_size=config.get('step_size', 10), gamma=config.get('gamma', 0.1))
    elif config['scheduler'] == 'multistep':
        scheduler = MultiStepLR(optimizer, milestones=config.get('milestones', [30, 60, 90]), gamma=config.get('gamma', 0.1))
    elif config['scheduler'] == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.get('gamma', 0.1),
                                      patience=config.get('plateau_patience', 10), verbose=True)
    elif config['scheduler'] == 'none':
        scheduler = None
    else:
        raise NotImplementedError(f"Scheduler {config['scheduler']} not implemented.")
    return scheduler

# utils/losses
# Standar Cross Entropy
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.ce(outputs, targets)

# Label Smoothing Cross Entropy (optional kalau mau)
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, outputs, targets):
        log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
