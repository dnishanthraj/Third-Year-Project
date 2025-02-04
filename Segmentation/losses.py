import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['BCEDiceLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        return 0.5 * bce + dice


class BCEDiceFocalLoss(nn.Module):
    """
    Combines Focal Loss + Dice Loss for binary segmentation.
    
    focal_weight: How strongly to weight the focal component vs. dice.
                  By default, we do 0.5 * focal + dice, but you can tune 
                  or make it a parameter.
    alpha, gamma: Standard Focal Loss hyperparameters.
    """
    def __init__(self, alpha=1.0, gamma=2.0, focal_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        """
        inputs: (N, 1, H, W) raw logits from model
        targets: (N, 1, H, W) binary ground-truth mask
        """
        # ------------------------------------------------
        # 1) Compute Focal Loss
        # ------------------------------------------------
        # BCE per-pixel, no reduction yet
        bce_per_pixel = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        # Convert BCE to pt = exp(-bce)
        pt = torch.exp(-bce_per_pixel)
        # Focal loss factor = alpha * (1-pt)^gamma
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_per_pixel

        # Average (or sum) over all pixels/batch
        focal_loss = focal_loss.mean()

        # ------------------------------------------------
        # 2) Compute Dice Loss
        # ------------------------------------------------
        smooth = 1e-5
        inputs_sigmoid = torch.sigmoid(inputs)
        num = targets.size(0)
        inputs_flat = inputs_sigmoid.view(num, -1)
        targets_flat = targets.view(num, -1)

        intersection = inputs_flat * targets_flat
        dice_score = (2. * intersection.sum(dim=1) + smooth) / (
            inputs_flat.sum(dim=1) + targets_flat.sum(dim=1) + smooth
        )
        dice_loss = 1 - dice_score.mean()

        # ------------------------------------------------
        # 3) Combine Focal and Dice
        # ------------------------------------------------
        # Example: 0.5 * focal + dice
        loss = self.focal_weight * focal_loss + dice_loss

        return loss
    
class FocalLoss(nn.Module):
    """
    Standard Focal Loss for binary segmentation/classification.
    
    alpha: weighting factor for positive examples (helps with class imbalance).
    gamma: focusing parameter that down-weights easy examples.
           Higher gamma places more focus on hard, misclassified examples.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: raw logits of shape (N, 1, H, W) or (N,) for binary classification
        targets: binary labels (same shape as inputs), 0 or 1
        """
        # 1) Compute binary cross-entropy for each pixel without reduction.
        bce_per_pixel = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 2) Convert BCE to pt = exp(-bce)
        pt = torch.exp(-bce_per_pixel)

        # 3) Focal Loss factor = alpha * (1 - pt)^gamma
        focal_term = self.alpha * (1 - pt) ** self.gamma

        # 4) Combine
        focal_loss = focal_term * bce_per_pixel

        # 5) Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
