import torch
from torch import nn

class RefineLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, refine_offsets, gt_offsets, reg_weights):
        num_pos = torch.nonzero(reg_weights > 0).size()[0]
        loss = self.smooth_l1_loss(refine_offsets, gt_offsets, reg_weights)
        if num_pos == 0:
            num_pos = 1.
        loss = loss.sum() / num_pos
        return loss

    def smooth_l1_loss(self, pred, gt, weights, beta=1. / 9):
        l1_loss = torch.abs(pred - gt)
        cond = l1_loss < beta
        loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
        loss = loss * weights
        return loss