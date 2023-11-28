import pytorch_model_summary as summary
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DenoisingBaseModel import DenoisingBaseModel


class FirstConv(nn.Module):
    """
    First convolutional layer of the the DnCNN model. The paper uses He initialization for the weights
    whereas the pytorch implementation shared by the authors uses orthogonal initialization.

    """

    def __init__(self, in_channels: int, output_channels: int, kernel_size: int):
        super().__init__()

        # Convolutional layer to upsample the feature map
        self.conv = nn.Conv1d(
            in_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,  # bias=True for Dncnn with some modifications
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        return self.relu(x)


class BodyConv(nn.Module):
    """
    ### Body convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        output_channels: int,
        kernel_size: int,
        eps: float,
        momentum: float,
    ):
        super().__init__()

        # Convolutional layer to upsample the feature map
        self.conv = nn.Conv1d(
            in_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

        self.batch_norm = nn.BatchNorm1d(
            output_channels, eps, momentum
        )  # added eps and momentum to batch norm

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.batch_norm(x)
        return self.relu(x)


class LastConv(nn.Module):
    """
    ### Last convolutional layer
    """

    def __init__(self, in_channels: int, output_channels: int, kernel_size: int):
        super().__init__()

        # Convolutional layer to upsample the feature map
        self.conv = nn.Conv1d(
            in_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class DnCNN(DenoisingBaseModel):
    """
    An implementation of the DnCNN architecture from the paper: Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising
    A link to the paper can be found here: https://arxiv.org/abs/1608.03981

    """

    def __init__(self, params) -> None:
        super().__init__(params)

        self.first_conv = FirstConv(
            in_channels=params.input_channels,
            output_channels=params.output_channels,
            kernel_size=params.kernel_size,
        )
        self.body_convs = nn.ModuleList(
            [
                BodyConv(
                    in_channels=params.output_channels,
                    output_channels=params.output_channels,
                    kernel_size=params.kernel_size,
                    eps=params.bn_eps,
                    momentum=params.bn_momentum,
                )
                for _ in range(params.depth - 2)
            ]
        )
        self.last_conv = LastConv(
            in_channels=params.output_channels,
            output_channels=params.input_channels,
            kernel_size=params.kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x  # save the input for the residual connection. this modification is part of Dncnn with some modifications
        x = self.first_conv(x)
        for conv in self.body_convs:
            x = conv(x)
        return y - self.last_conv(x)
