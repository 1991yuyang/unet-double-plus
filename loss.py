import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, targets):
        inputs = self.softmax(inputs)[:, 1, :, :]
        targets = targets.type(torch.FloatTensor).cuda(0)
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, output, target):
        target = target.type(torch.FloatTensor).cuda(0)
        softmax_result = self.softmax(output)[:, 1, :, :]  # N * H * W
        numerator = 2 * torch.sum(softmax_result * target, dim=[1, 2]) + 1
        denominator = torch.sum(torch.pow(softmax_result, 2), dim=[1, 2]) + torch.sum(torch.pow(target, 2), dim=[1, 2]) + 1
        loss = torch.mean(1.0 - numerator / denominator)
        return loss


class DiceLoss_CeLoss(nn.Module):

    def __init__(self, dice_loss_weight, ce_loss_weight):
        """
        weighted add dice loss and crossentropy loss
        """
        super(DiceLoss_CeLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.dice_loss_weight = dice_loss_weight
        self.ce_loss_weight = ce_loss_weight

    def forward(self, output, target):
        ce_loss = self.ce(output, target)
        dice_loss = self.dice(output, target)
        loss = self.dice_loss_weight * dice_loss + self.ce_loss_weight * ce_loss
        return loss