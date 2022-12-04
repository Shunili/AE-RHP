import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
 
class HLBLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, ce_ignore_index, hlb_ignore_index, loss_weight_ce=1, loss_weight_hlb=1,reduction='mean'):
        super(HLBLoss, self).__init__()
        self.ce_ignore_index = ce_ignore_index
        self.hlb_ignore_index = hlb_ignore_index
        self.loss_weight_hlb = loss_weight_hlb
        self.loss_weight_ce = loss_weight_ce
        self.reduction = reduction


    def mse_loss(self, input, target, ignored_index):
        mask = target == ignored_index
        out = (input[~mask]-target[~mask])**2
        if self.reduction == "mean":
            return out.mean()
        elif self.reduction == "None":
            return out

    def forward(self, pred_input, target, pred_hlb, true_hlb):

        ce_loss = F.cross_entropy(pred_input, target, reduction=self.reduction, ignore_index = self.ce_ignore_index)
        regression_loss = self.mse_loss(pred_hlb, true_hlb, ignored_index = self.hlb_ignore_index)

        loss = self.loss_weight_ce * ce_loss + self.loss_weight_hlb*regression_loss
        losses = OrderedDict(loss=loss, recon_loss=ce_loss, hlb_loss=regression_loss)

        return losses

def get_loss(name, **kwargs):
    """Factory function for loss functions"""

    # First, look for local custom loss
    try:
        Loss = globals()[name]

    # Then try to search PyTorch loss functions
    except KeyError:
        Loss = getattr(torch.nn, name)

    return Loss(**kwargs)