import torch
import torch.nn as nn
from torch.nn.functional import relu
import math
import os
import yaml
import torch.nn.functional as F
from models.DenoisingBaseModel import DenoisingBaseModel

import wandb


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
        

class DenoisingAutoencoderTransients(DenoisingBaseModel):
    """Denoising autoencoder with transposed convolution.

    General architecture taken from: Medical image denoising using convolutional
    denoising autoencoders by Lovedeep Gondara:
    https://ieeexplore.ieee.org/document/7836672/?arnumber=7836672

    Args:
        params: Parameters for the model. See DenoisingBaseModel for more info.
            The parameters are passed through the wandb config file.


    Returns:
        torch.Tensor: The denoised spectra.

    """

    def __init__(self, params):
        super().__init__(params)

        self.cyclic_padding = CyclicPaddingConv(in_channels=2, out_channels=2)

        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 2),
            stride=(1, 2),
        )
        self.t_conv2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=2,
            kernel_size=(1, 2),
            stride=(1, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): The input spectra.

        Returns:
            torch.Tensor: The denoised spectra.

        """
        x = self.cyclic_padding(x)
        x = relu(self.conv1(x))
        x = self.pool(x)
        x = relu(self.conv2(x))
        x = self.pool(x)
        x = relu(self.conv2(x))
        x = relu(self.t_conv1(x))
        x = self.t_conv2(x)

        return x
