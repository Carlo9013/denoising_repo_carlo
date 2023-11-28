""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DenoisingBaseModel import DenoisingBaseModel


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        mid_channels: int = None,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bilinear=True,
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                mid_channels=in_channels // 2,
            )
        else:
            self.up = nn.ConvTranspose1d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


""" Full assembly of the parts to form the complete network """


class UNet(DenoisingBaseModel):
    """UNet model ""

    Args:
        params (dict): Dictionary containing the parameters of the model
        bilinear (bool, optional): Whether to use bilinear interpolation or transposed convolutions. Defaults to False.
    """

    def __init__(self, params, bilinear=False):
        super().__init__(params)
        self.bilinear = bilinear

        self.inc = DoubleConv(
            in_channels=params.input_channels,
            out_channels=64,
            kernel_size=params.kernel_size,
        )
        self.down1 = Down(64, 128, params.kernel_size)
        self.down2 = Down(128, 256, params.kernel_size)
        self.down3 = Down(256, 512, params.kernel_size)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, params.kernel_size)
        self.up1 = Up(1024, 512 // factor, params.kernel_size, bilinear)
        self.up2 = Up(512, 256 // factor, params.kernel_size, bilinear)
        self.up3 = Up(256, 128 // factor, params.kernel_size, bilinear)
        self.up4 = Up(128, 64, params.kernel_size, bilinear)
        self.outc = OutConv(in_channels=64, out_channels=params.input_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
