# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

"""
Loss functions used in TruFor training.

This module contains all the loss functions used for training TruFor model,
including localization, confidence, and detection losses.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class CrossEntropy(nn.Module):
    """Cross-entropy loss with optional class weights and ignore label."""
    
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)

    def forward(self, score, target):        
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)
        return loss

    
class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    
    def __init__(self, ignore_label=-1, smooth=1, exponent=2):
        super(DiceLoss, self).__init__()  
        self.ignore_index = ignore_label
        self.smooth = smooth
        self.exponent = exponent
        
    def dice_loss(self, pred, target, valid_mask):
        assert pred.shape[0] == target.shape[0]
        total_loss = 0
        num_classes = pred.shape[1]
        for i in range(num_classes):
            if i != self.ignore_index:
                dice_loss = self.binary_dice_loss(
                    pred[:, i],
                    target[..., i],
                    valid_mask=valid_mask,)
                total_loss += dice_loss
        return total_loss / num_classes

    def binary_dice_loss(self, pred, target, valid_mask):
        assert pred.shape[0] == target.shape[0]
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

        num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + self.smooth
        den = torch.sum(pred.pow(self.exponent)*valid_mask + target.pow(self.exponent)*valid_mask, dim=1) + max(self.smooth, 1e-5)
        
        dice = num / den
        dice = torch.mean(dice)
        return 1 - dice
        
    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')
        
        score = F.softmax(score,dim=1)
        num_classes = score.shape[1]
        
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()
        
        loss = self.dice_loss(score, one_hot_target, valid_mask)
        return loss


class BinaryDiceLoss(nn.Module):
    """Binary Dice loss for binary segmentation."""
    
    def __init__(self, smooth=1, exponent=2, ignore_label=-1):
        super(BinaryDiceLoss, self).__init__()  
        self.ignore_index = ignore_label
        self.smooth = smooth
        self.exponent = exponent

    def binary_dice_loss(self, pred, target, valid_mask):
        assert pred.shape[0] == target.shape[0]
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

        num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + self.smooth
        den = torch.sum(pred.pow(self.exponent)*valid_mask + target.pow(self.exponent)*valid_mask, dim=1) + max(self.smooth, 1e-5)
        
        dice = num / den
        dice = torch.mean(dice)
        return 1 - dice
        
    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')
        
        score = F.softmax(score,dim=1)
        num_classes = score.shape[1]
        
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()
        
        loss = self.binary_dice_loss(
                    score[:, 1],
                    one_hot_target[..., 1],
                    valid_mask)
        return loss


class DiceEntropyLoss(nn.Module):
    """Combined Dice and Cross-entropy loss (main loss used in TruFor)."""
    
    def __init__(self, smooth=1, exponent=2, ignore_label=-1, weight=None):
        super(DiceEntropyLoss, self).__init__()  
        self.ignore_label = ignore_label
        self.smooth = smooth
        self.exponent = exponent
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)
    
    def binary_dice_loss(self, pred, target, valid_mask):
        assert pred.shape[0] == target.shape[0]
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

        num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + self.smooth
        den = torch.sum(pred.pow(self.exponent)*valid_mask + target.pow(self.exponent)*valid_mask, dim=1) + max(self.smooth, 1e-5)
        
        dice = num / den
        dice = torch.mean(dice)
        return 1 - dice
        
    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')
        
        CE_loss   = self.cross_entropy(score, target)
        
        score = F.softmax(score,dim=1)
        num_classes = score.shape[1]
        
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_label).long()
        
        dice_loss = self.binary_dice_loss(
                    score[:, 1],
                    one_hot_target[..., 1],
                    valid_mask)

        return 0.3*CE_loss + 0.7*dice_loss


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2., ignore_label=-1):
        super(FocalLoss, self).__init__()
        self.alpha=alpha
        self.gamma= gamma
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction="none")
       
    def forward(self, score, target):  
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')
            
        ce_loss = self.criterion(score, target)
        pt = torch.exp(-ce_loss)
        f_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return f_loss.mean()


class MSE(nn.Module):
    """MSE loss for confidence estimation."""
    
    def __init__(self, ignore_label=-1, criterion='mse'):
        super(MSE, self).__init__()
        self.ignore_label = ignore_label
        if criterion=='mse':
            self.criterion = nn.MSELoss()
        else:
            assert False
    
    def calcolaGTs(self, gt, erodeKernSize=15, dilateKernSize=11):
        from torch.nn.functional import max_pool2d
        gt1 = 1 - max_pool2d(1-gt[:,None,:,:], erodeKernSize, stride=1, padding=(erodeKernSize-1)//2)[:,0]
        gt0 = 1 - max_pool2d(gt[:,None,:,:], dilateKernSize, stride=1, padding=(dilateKernSize-1)//2)[:,0]
        return gt0, gt1

    def forward(self, pred, target, conf):
        # conf: confidence prediction (1 channel)
        # pred: 2 channels cmx prediction
        ch, cw = conf.size(2), conf.size(3)
        ph, pw = pred.size(2), pred.size(3)
        h, w = target.size(1), target.size(2)
        
        if ph != h or pw != w:
            pred = F.upsample(input=pred, size=(h, w), mode='bilinear')
        if ch != h or cw != w:
            conf = F.upsample(input=conf, size=(h, w), mode='bilinear')
        
        conf = torch.sigmoid(conf)
        pred = F.softmax(pred, dim=1)
        
        target0, target1 = self.calcolaGTs((target==1).float())
        conf = conf.squeeze(dim=1)
        tcp  = pred[:,1]*target1 + pred[:,0]*target0
        
        assert conf.shape == tcp.shape
        
        valid = torch.logical_and(target!=self.ignore_label, torch.logical_or(target1>0, target0>0))
                
        conf = conf[valid]
        tcp  = tcp[valid]
        loss = self.criterion(conf, tcp)
        return loss


class DetectionCrossEntropy(nn.Module):
    """Cross-entropy loss for detection task."""
    
    def __init__(self):
        super(DetectionCrossEntropy, self).__init__()

    def forward(self, score, target):
        target_det = (torch.count_nonzero(target * (target >= 0), (-1, -2)) > 3).float().clamp(0, 1)
        weights_det = target_det * 0.5 / 0.7 + (1 - target_det) * 0.5 / 0.3
        loss_det = F.binary_cross_entropy_with_logits(score[:, 0], target_det, reduction='mean', weight=weights_det)
        return loss_det


# Aliases for backward compatibility
CrossEntropyLoss = CrossEntropy
