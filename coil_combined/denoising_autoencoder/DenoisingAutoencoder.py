import torch
import torch.nn as nn
from torch.nn.functional import relu

from models.DenoisingBaseModel import DenoisingBaseModel


class DenoisingAutoencoder(DenoisingBaseModel):
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

        self.conv1 = nn.Conv1d(
            in_channels=params.input_channels,
            out_channels=params.output_channels,
            kernel_size=params.kernel_size,
            padding=params.kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=params.output_channels,
            out_channels=params.output_channels,
            kernel_size=params.kernel_size,
            padding=params.kernel_size // 2,
        )
        self.pool = nn.MaxPool1d(2, 2)
        # Decoder
        self.t_conv1 = nn.ConvTranspose1d(
            in_channels=params.output_channels,
            out_channels=params.output_channels,
            kernel_size=2,
            stride=2,
        )
        self.t_conv2 = nn.ConvTranspose1d(
            in_channels=params.output_channels,
            out_channels=params.input_channels,
            kernel_size=2,
            stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): The input spectra.

        Returns:
            torch.Tensor: The denoised spectra.

        """
        x = relu(self.conv1(x))
        x = self.pool(x)
        x = relu(self.conv2(x))
        x = self.pool(x)
        x = relu(self.conv2(x))
        x = relu(self.t_conv1(x))
        x = self.t_conv2(x)

        return x
