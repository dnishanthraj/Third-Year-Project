# ---------------------------------------------------------
# U-Net Model Definition
# ---------------------------------------------------------

import torch.nn.functional as F
from .unet_parts import *  # Import building blocks for U-Net

class UNet(nn.Module):
    """
    Standard U-Net architecture for image segmentation.

    Args:
        n_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
        n_classes (int): Number of output segmentation classes.
        bilinear (bool): Whether to use bilinear upsampling (True) or transposed convolutions (False).
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder path (contracting path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1  # Adjustment if bilinear is used
        self.down4 = Down(512, 1024 // factor)

        # Decoder path (expanding path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output convolution
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """
        Defines the forward pass of the U-Net model.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits
