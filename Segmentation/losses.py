# ---------------------------------------------------------
# Loss Functions for Segmentation Training
# ---------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['BCEDiceLoss', 'BCEDiceFocalLoss', 'FocalLoss']

# ------------------------
# BCE + Dice Combined Loss
# ------------------------

class BCEDiceLoss(nn.Module):
    """
    Combines Binary Cross-Entropy (BCE) and Dice Loss.
    Suitable for binary segmentation tasks.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        """
        Args:
            input (torch.Tensor): Raw logits from model.
            target (torch.Tensor): Binary ground-truth masks.

        Returns:
            torch.Tensor: Combined BCE + Dice loss value.
        """
        # Binary Cross-Entropy Loss
        bce = F.binary_cross_entropy_with_logits(input, target)

        # Dice Loss
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        # Final combined loss
        return 0.5 * bce + dice

# ------------------------
# BCE + Dice + Focal Combined Loss
# ------------------------

class BCEDiceFocalLoss(nn.Module):
    """
    Combines Focal Loss and Dice Loss for binary segmentation tasks.

    Args:
        alpha (float): Weighting factor for positive examples.
        gamma (float): Focusing parameter for Focal Loss.
        focal_weight (float): Scaling factor for Focal loss contribution.
    """
    def __init__(self, alpha=1.0, gamma=2.0, focal_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Raw logits from model.
            targets (torch.Tensor): Binary ground-truth masks.

        Returns:
            torch.Tensor: Combined Focal + Dice loss value.
        """
        # --------------------
        # Focal Loss Computation
        # --------------------
        bce_per_pixel = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_per_pixel)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_per_pixel
        focal_loss = focal_loss.mean()

        # --------------------
        # Dice Loss Computation
        # --------------------
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

        # --------------------
        # Final Loss
        # --------------------
        loss = self.focal_weight * focal_loss + dice_loss

        return loss

# ------------------------
# Standard Focal Loss
# ------------------------

class FocalLoss(nn.Module):
    """
    Standard Focal Loss for binary segmentation or classification.

    Args:
        alpha (float): Balancing factor for positive/negative classes.
        gamma (float): Focusing parameter to down-weight easy examples.
        reduction (str): Reduction method: 'mean', 'sum', or 'none'.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Raw logits (shape: (N, 1, H, W) or (N,))
            targets (torch.Tensor): Binary ground-truth masks or labels.

        Returns:
            torch.Tensor: Focal loss value.
        """
        # Binary Cross-Entropy Loss per pixel
        bce_per_pixel = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # pt = probability of correct classification
        pt = torch.exp(-bce_per_pixel)

        # Focal loss factor
        focal_term = self.alpha * (1 - pt) ** self.gamma

        # Final focal loss
        focal_loss = focal_term * bce_per_pixel

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
