# ---------------------------------------------------------
# Utility Functions and Classes
# ---------------------------------------------------------

import argparse

def str2bool(v):
    """
    Converts a string to a boolean value for argument parsing.

    Args:
        v (str): Input string ('true', 'false', '1', '0').

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the input is not a valid boolean string.
    """
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def count_params(model):
    """
    Counts the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): The model whose parameters are to be counted.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AverageMeter(object):
    """
    Computes and stores the average and current value of a metric.
    Useful for tracking loss, accuracy, Dice, IoU, etc. during training/validation.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets all attributes to zero.
        """
        self.val = 0  # Current value
        self.avg = 0  # Running average
        self.sum = 0  # Cumulative sum
        self.count = 0  # Number of updates

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): New observed value.
            n (int, optional): Weight for the update (e.g., batch size). Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
