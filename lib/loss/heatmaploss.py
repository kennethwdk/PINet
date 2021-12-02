import torch
from torch import nn

class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        loss = ((pred - gt)**2) * mask
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)
        return loss