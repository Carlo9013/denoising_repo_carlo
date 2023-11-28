import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from wandb.sdk.wandb_config import Config as wandbConfig
from models.DenoisingBaseModel import DenoisingBaseModel
from CBAM import CBAM2d
import math


# U-Net code adapted from: https://github.com/milesial/Pytorch-UNet
class CyclicPadding2D(nn.Module):
    def __init__(self, n=8, dim=-2):
        super().__init__()
        assert n > 1
        if n > 1:
            self.n1, self.n2 = int(math.floor(n / 2)), int(math.ceil(n / 2))
            if n % 2 == 0:
                self.n2 += 1
        else:
            self.n1 = False
        self.dim = dim

    def forward(self, input):
        # Input.shape: [bS, channels, num_coil, spectral_length)
        # Can use real/imag channels or one complex channel
        if not self.n1:
            return input
        else:
            return torch.cat(
                [input[..., self.n2 :, :], input, input[..., : self.n1, :]],
                dim=self.dim,
            )


class CyclicPaddingConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_coils: int = 8,
        y: int = 3,
        n=None,
    ):
        super().__init__()
        if n is None:
            n = math.floor(y / 2)
        self.cyclic_padding = CyclicPadding2D(n=num_coils)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(num_coils, y),
            stride=1,
            padding=(0, n),
        )

    def forward(self, x):
        x = self.cyclic_padding(x)
        x = self.conv1(x)
        return x


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        mid_channels=None,
        residual=False,
    ):
        super().__init__()
        self.residual = residual
        self.kernel_size = kernel_size

        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            DoubleConv(in_channels, in_channels, kernel_size, residual=True),
            DoubleConv(in_channels, out_channels, kernel_size),
        )

        # //TODO: Double conv with cyclic padding and assymterical kernel

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=(1, 2))
            self.conv = DoubleConv(in_channels, in_channels, kernel_size, residual=True)
            self.conv2 = DoubleConv(
                in_channels, out_channels, kernel_size, in_channels // 2
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels, kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.conv2(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CBAMAttentionUnetTransients(DenoisingBaseModel):
    """Architecture based on U-Net with self-attention

    Args:
        nn (nn.Module): Base class for all neural network modules
    """

    def __init__(self, params: wandbConfig, bilinear=True, **kwargs):
        super().__init__(params)
        self.bilinear = bilinear
        self.kernel_size = params.kernel_size

        self.cyclic_padding = CyclicPaddingConv(in_channels=2, out_channels=2)
        self.inc = DoubleConv(params.input_channels, 64, self.kernel_size)
        self.down1 = Down(64, 128, self.kernel_size)
        self.down2 = Down(128, 256, self.kernel_size)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor, self.kernel_size)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, params.input_channels)
        self.sa1 = CBAM2d(256)
        self.sa2 = CBAM2d(256)
        self.sa3 = CBAM2d(128)

    def forward(
        self,
        x,
    ) -> torch.Tensor:
        """Model is U-Net with a self-attention module.

        The only difference between this model and the one used for diffusion, is
        that this model does not have positional encodings for the time step. The
        rest of the architecture is the same.

        Args:
            x (torch.Tensor): Input data or spectrogram.
        """
        x = self.cyclic_padding(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x3 = self.sa1(x3)

        x4 = self.down3(x3)
        x4 = self.sa2(x4)

        x = self.up1(x4, x3)

        x = self.sa3(x)

        x = self.up2(x, x2)
        x = self.up3(x, x1)
        output = self.outc(x)
        return output
