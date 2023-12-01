# https://blog.paperspace.com/attention-mechanisms-in-computer-vision-cbam/#:~:text=Spatial%20attention%20represents%20the%20attention,features%20that%20define%20that%20bird.

from collections import OrderedDict

import torch
import torch.nn as nn
from typing import Union, Tuple
from typing import Optional, Tuple, Union, List

__all__ = [
    "citation",
    "ChannelGate",
    "SpatialGate",
    "EfficientChannelAttention",
    "CBAM1d",
    "CBAM2d",
    "CBAM3d",
]


citation = OrderedDict(
    {
        "CBAM": {
            "Title": "CBAM: Convolutional Block Attention Module",
            "Authors": "Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon",
            "Year": "2018",
            "Journal": "ECCV",
            "Institution": "Korea Advanced Institute of Science and Technology, Lunit Inc., and Adobe Research",
            "URL": "https://arxiv.org/pdf/1807.06521.pdf",
            "Notes": "Added the possiblity to switch from SE to eCA in the ChannelGate and updated deprecated sigmoid",
            "Source Code": [
                "Modified from: https://github.com/Jongchan/attention-module",
                "Implemented by: Ing. John LaMaster",
            ],
        },
        "ECA": {
            "Title": "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks",
            "Authors": "Qilong Wang, Banggu Wu, Pengfei Zhu, Peihua Li, Wangmeng Zuo, Qunghua Hu",
            "Year": 2020,
            "Journal": "CVPR",
            "Institutions": [
                "Tianjin Key Lab of Machine Learning, College of Intelligence and Computing, Tianjin University, China",
                "Dalian University of Technology, China",
                "Harbin Institute of Technology, China",
            ],
            "Source Code": "Written by Ing. John LaMaster",
        },
    }
)


class BasicConv(nn.Module):
    no_dropout = True

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: int = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: int = 1,
        groups: int = 1,
        relu: bool = True,
        bn: bool = True,
        bias: bool = False,
        dim: int = 1,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.dim = dim
        self.conv = get_conv_layer(
            dim,
            "conv",
            in_planes,
            out_planes,
            kernel_size,
            groups=groups,
            stride=stride,
            pad=padding,
            bias=bias,
        )
        if dim == 1:
            self.bn = (
                nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01, affine=True)
                if bn
                else None
            )
        elif dim == 2:
            self.bn = (
                nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
                if bn
                else None
            )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class EfficientChannelAttention(nn.Module):
    no_dropout = True

    def __init__(
        self,
        num_channels: int,
        gamma: int = 2,  # fixed value
        b: int = 1,  # fixed value
        dim: int = 1,
    ):  # data: 1d vs 2d vs 3d
        self.dim = dim
        super(EfficientChannelAttention, self).__init__()
        t = int(
            torch.abs(
                (torch.log2(torch.tensor(num_channels, dtype=torch.float64)) + b)
                / gamma
            )
        )
        k = t if t % 2 else t + 1

        self.conv = get_conv_layer(
            dim, "conv", 1, 1, kernel=k, stride=1, pad=int(k / 2), bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            out = x.transpose(-1, -2)
            out = self.conv(out)
            out = out.transpose(-1, -2)
        else:
            out = x.transpose(-1, -3)
            out = self.conv(out)
            out = out.transpose(-1, -3)
        return out


class ChannelGate(nn.Module):
    no_dropout = True

    def __init__(
        self,
        gate_channels: int,
        reduction_ratio: int = 16,
        pool_types: list = ["avg", "max"],
        method: str = "efficient",
        dim: int = 1,
        offset: float = 0.0,
    ):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.avg_pool = get_adaptive_pooling_layer(dim=dim, type="avg", size=1)
        self.max_pool = get_adaptive_pooling_layer(dim=dim, type="max", size=1)
        if method == "efficient":
            self.attention = EfficientChannelAttention(gate_channels, dim=dim)
        elif method == "mlp":
            self.attention = nn.Sequential(
                nn.Flatten(),
                nn.Linear(gate_channels, gate_channels // reduction_ratio),
                nn.ReLU(),
                nn.Linear(gate_channels // reduction_ratio, gate_channels),
            )
        self.pool_types = pool_types
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = self.avg_pool(x)
                channel_att_raw = self.attention(avg_pool)
            elif pool_type == "max":
                max_pool = self.max_pool(x)
                channel_att_raw = self.attention(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = (
                    torch.sigmoid(channel_att_sum + channel_att_raw) + self.offset
                )

        scale = (
            channel_att_sum.expand_as(x)
            if channel_att_sum.dim() >= 3
            else channel_att_sum.unsqueeze(-1).expand_as(x)
        )
        return x * scale


class ChannelPool(nn.Module):
    no_dropout = True

    def forward(
        self,
        x: torch.Tensor,  # torch.Size([bS, num_channels, ...])
    ) -> torch.Tensor:
        return torch.cat(
            (torch.amax(x, dim=1, keepdim=True), torch.mean(x, dim=1, keepdim=True)),
            dim=1,
        )


class SpatialGate(nn.Module):
    no_dropout = True

    def __init__(
        self, dim: int, kernel_size: int = 7, offset: float = 0.0  # (int,tuple)=7,
    ):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2,
            1,
            kernel_size,
            stride=1,
            relu=False,
            padding=(kernel_size - 1) // 2,
            dim=dim,
        )
        self.sigmoid = nn.Sigmoid()
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out) + self.offset
        return x * scale


class CBAM1d(nn.Module):
    no_dropout = True

    def __init__(
        self,
        gate_channels: int,
        reduction_ratio: int = 16,
        pool_types: list = ["avg", "max"],
        no_spatial: bool = False,
        no_channel: bool = False,
        offset: float = 0.0,
    ):
        super(CBAM1d, self).__init__()
        if not no_channel:
            self.ChannelGate = ChannelGate(
                gate_channels, reduction_ratio, pool_types, dim=1, offset=offset
            )
        else:
            self.ChannelGate = nn.Identity()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(dim=1, offset=offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class CBAM1dTimeEmbedding(nn.Module):
    no_dropout = True

    def __init__(
        self,
        gate_channels: int,
        reduction_ratio: int = 16,
        pool_types: list = ["avg", "max"],
        no_spatial: bool = False,
        no_channel: bool = False,
        offset: float = 0.0,
    ):
        super(CBAM1dTimeEmbedding, self).__init__()
        if not no_channel:
            self.ChannelGate = ChannelGate(
                gate_channels, reduction_ratio, pool_types, dim=1, offset=offset
            )
        else:
            self.ChannelGate = nn.Identity()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(dim=1, offset=offset)

    def forward(
        self, x: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class CBAM2d(nn.Module):
    no_dropout = True

    def __init__(
        self,
        gate_channels: int,
        reduction_ratio: int = 16,
        pool_types: list = ["avg", "max"],
        no_spatial: bool = False,
        offset: float = 0.0,
        kernel_size: int = 7,  # (int,tuple)=7, # tuple(7,1)
    ):
        super(CBAM2d, self).__init__()
        self.ChannelGate = ChannelGate(
            gate_channels, reduction_ratio, pool_types, dim=2, offset=offset
        )
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(
                dim=2, offset=offset, kernel_size=kernel_size
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class CBAM3d(nn.Module):
    no_dropout = True

    def __init__(
        self,
        gate_channels: int,
        reduction_ratio: int = 16,
        pool_types: list = ["avg", "max"],
        no_spatial: bool = False,
        offset: float = 0.0,
    ):
        super(CBAM3d, self).__init__()
        self.ChannelGate = ChannelGate(
            gate_channels, reduction_ratio, pool_types, dim=3, offset=offset
        )
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(dim=3, offset=offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


################################################################################
############################# Auxiliary Functions ##############################
################################################################################


def get_conv_layer(
    dim: int,
    net: str = "conv",
    in_c: Union[int, float] = 1,
    out_c: Union[int, float] = 1,
    kernel: int = 3,
    groups: int = 1,
    stride: int = 1,
    pad: int = 0,
    pad_o: int = 0,
    bias: bool = False,
):
    in_c = int(in_c)
    out_c = int(out_c)
    if dim == 1 and net == "conv":
        return nn.Conv1d(
            in_c,
            out_c,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            bias=bias,
            groups=groups,
        )
    elif dim == 1 and net == "trans":
        return nn.ConvTranspose1d(
            in_c,
            out_c,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            output_padding=pad_o,
            bias=bias,
            groups=groups,
        )
    elif dim == 2 and net == "conv":
        return nn.Conv2d(
            in_c,
            out_c,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            bias=bias,
            groups=groups,
        )
    elif dim == 2 and net == "trans":
        return nn.ConvTranspose2d(
            in_c,
            out_c,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            output_padding=pad_o,
            bias=bias,
            groups=groups,
        )
    elif dim == 3 and net == "conv":
        return nn.Conv3d(
            in_c,
            out_c,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            bias=bias,
            groups=groups,
        )
    elif dim == 3 and net == "trans":
        return nn.ConvTranspose3d(
            in_c,
            out_c,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            output_padding=pad_o,
            bias=bias,
            groups=groups,
        )
    else:
        raise ValueError("Input data dimensionality must be between 1D and 3D")


def get_adaptive_pooling_layer(dim: int = 1, type: str = "avg", size: int = 1):
    if dim == 1 and "avg" in type:
        return nn.AdaptiveAvgPool1d(size)
    elif dim == 1 and "max" in type:
        return nn.AdaptiveMaxPool1d(size)
    elif dim == 2 and "avg" in type:
        return nn.AdaptiveAvgPool2d(size)
    elif dim == 2 and "max" in type:
        return nn.AdaptiveMaxPool2d(size)
    elif dim == 3 and "avg" in type:
        return nn.AdaptiveAvgPool3d(size)
    elif dim == 3 and "max" in type:
        return nn.AdaptiveMaxPool3d(size)
    else:
        raise ValueError("Input data dimensionality must be between 1D and 3D")


def get_norm_layer(dim, norm, arg, **kwargs):
    assert dim == 1 or dim == 2 or dim == 3
    if dim == 1:
        if norm == "instance":
            return [nn.InstanceNorm1d(arg)]
        elif norm == "batch":
            return nn.BatchNorm1d(arg, kwargs)
    elif dim == 2:
        if norm == "instance":
            return [nn.InstanceNorm2d(arg)]
        elif norm == "batch":
            return [nn.BatchNorm2d(arg, kwargs)]
    elif dim == 3:
        if norm == "instance":
            return [nn.InstanceNorm3d(arg)]
        elif norm == "batch":
            return [nn.BatchNorm3d(arg, kwargs)]
    else:
        raise ValueError("Input data dimensionality must be between 1D and 3D")


if __name__ == "__main__":
    input = torch.randn(1, 2, 8, 512)
    cbam = CBAM2d(2)
    output = cbam(input)
    print(output.shape)
