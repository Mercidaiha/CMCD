# coding: utf-8
# 2021/6/19 @ tongshiwei

import torch
from torch import nn


# class PairSCELoss(nn.Module):
#     def __init__(self, reduction="mean"):
#         super(PairSCELoss, self).__init__()
#         self._loss = nn.CrossEntropyLoss(reduction=reduction)

#     def forward(self, pred_theta, pred_theta_pair, pos, *args):
#         """
#         pos is either 1 or -1
#         could be seen as predicting the sign based on the pred_theta and pred_theta_pair
#         1: pred_theta should be greater than pred_theta_pair
#         -1: otherwise
#         """
#         pred_theta = pred_theta.mean(dim=1)
#         pred_theta_pair = pred_theta_pair.mean(dim=1)
        
#         pred = torch.stack([pred_theta, pred_theta_pair], dim=1)
#         # print(pred)
#         # print(pos)
#         # print(((torch.ones(pred_theta.shape[0], device=pred.device) - pos) / 2).long())
        
#         return self._loss(pred, ((torch.ones(pred_theta.shape[0], device=pred.device) - pos) / 2).long())


import torch
import torch.nn as nn

class PairSCELoss(nn.Module):
    def __init__(self):
        super(PairSCELoss, self).__init__()

    def forward(self, pred_theta, pred_theta_pair, pos, *args):
        """
        pos is either -1.0 or 1.0
        could be seen as predicting the sign based on the pred_theta and pred_theta_pair
        1.0: pred_theta should be greater than pred_theta_pair
        -1.0: pred_theta should be less than pred_theta_pair
        """
        # print("pred_theta.shape")
        # print(pred_theta.shape)
        if pred_theta.dim() == 1:
            pred_theta = pred_theta.unsqueeze(-1)
        if pred_theta_pair.dim() == 1:
            pred_theta_pair = pred_theta_pair.unsqueeze(-1)
        # print(pred_theta.shape)
            
        pred_theta = pred_theta.mean(dim=1)
        pred_theta_pair = pred_theta_pair.mean(dim=1)
        
        if pred_theta.dim() == 1:
            pred_theta = pred_theta.unsqueeze(-1)
        if pred_theta_pair.dim() == 1:
            pred_theta_pair = pred_theta_pair.unsqueeze(-1)

        loss = torch.where(
            ((pos == 1.0) & (pred_theta > pred_theta_pair)) | (pos == -1.0),
            torch.zeros_like(pred_theta),
            (pred_theta - pred_theta_pair) ** 2
        )
        
        return loss.mean()  # you can change this to .sum() if you want a total loss instead of average



class HarmonicLoss(object):
    def __init__(self, zeta: (int, float) = 0.):
        self.zeta = zeta

    def __call__(self, score_loss, theta_loss, *args, **kwargs):
        # return ((1 - self.zeta) * score_loss + self.zeta * theta_loss)
        return (score_loss + self.zeta * theta_loss)