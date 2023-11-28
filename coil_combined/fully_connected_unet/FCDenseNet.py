"""
This is an implementation of the fully connected DenseNet architecture.
It is based on the paper "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation"
The code has been adapted from the following GtiHub repository: bfortuner/pytorch_tiramisu
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_model_summary import summary

from models.DenoisingBaseModel import DenoisingBaseModel


class DenseLayer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
    ):
        super().__init__()
        self.add_module("norm", nn.BatchNorm1d(in_channels))
        self.add_module("relu", nn.ReLU(True))
        self.add_module(
            "conv",
            nn.Conv1d(
                in_channels,
                growth_rate,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True,
            ),
        )
        self.add_module("drop", nn.Dropout1d(0.2))

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList(
            [
                DenseLayer(
                    in_channels=in_channels + i * growth_rate,
                    growth_rate=growth_rate,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, x):
        if self.upsample:
            new_features = []
            # we pass all previous activations into each dense layer normally
            # But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features, 1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)  # 1 = channel axis
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.add_module("norm", nn.BatchNorm1d(num_features=in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv1d(
                in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )
        self.add_module("drop", nn.Dropout1d(0.2))
        self.add_module("maxpool", nn.MaxPool1d(2))

    def forward(self, x):
        return super().forward(x)


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=True,
        )

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.add_module(
            "bottleneck", DenseBlock(in_channels, growth_rate, n_layers, upsample=True)
        )

    def forward(self, x):
        return super().forward(x)


def center_crop(layer, max_height):
    _, _, h = layer.size()
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2 : (xy2 + max_height)]


class FCDenseNet(DenoisingBaseModel):
    def __init__(
        self,
        params,
        down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4),
        bottleneck_layers=4,
        growth_rate=12,
        out_chans_first_conv=48,
    ):
        super().__init__(params)
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []
        in_channels = params.input_channels

        ## First Convolution ##

        self.firstconv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_chans_first_conv,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i])
            )
            cur_channels_count += growth_rate * down_blocks[i]
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.bottleneck = Bottleneck(cur_channels_count, growth_rate, bottleneck_layers)

        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(
                TransitionUp(prev_block_channels, prev_block_channels)
            )
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(
                DenseBlock(cur_channels_count, growth_rate, up_blocks[i], upsample=True)
            )
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##

        self.transUpBlocks.append(
            TransitionUp(prev_block_channels, prev_block_channels)
        )
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(
            DenseBlock(cur_channels_count, growth_rate, up_blocks[-1], upsample=False)
        )
        cur_channels_count += growth_rate * up_blocks[-1]

        ## Reconstruction Layer ##

        self.finalConv = nn.Conv1d(
            in_channels=cur_channels_count,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        return out


if __name__ == "__main__":
    input = torch.rand(1, 2, 512)
    model = FCDenseNet(
        in_channels=2,
        down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4),
        bottleneck_layers=4,
        growth_rate=12,
        out_chans_first_conv=48,
    )

    output = model(input)
    print(output.shape)
