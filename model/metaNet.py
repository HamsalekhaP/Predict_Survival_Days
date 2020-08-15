
# standard imports
import math

# other library imports
import torch
from torch import nn as nn
import torch.nn.functional as F

# local imports
import config as CONFIG


# Metastases prediction architecture with 54 sized MRI sub-volume + age as input
class MetaNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=8):
        super().__init__()
        if CONFIG.USE_AS_CHANNELS:
            in_channels = 4
        self.initBatchNorm = nn.BatchNorm3d(in_channels)
        self.block1 = ConvBlock(in_channels, out_channels, True)  # 25
        self.block2 = ConvBlock(out_channels, out_channels * 2, True)  # 10
        self.block3 = ConvBlock(out_channels * 2, out_channels * 4, False)  # 6
        self.block4 = ConvBlock(out_channels * 4, out_channels * 4, False)  # 2
        self.fc = nn.Linear(256 + CONFIG.FEED_AGE, 16)
        self.fc2 = nn.Linear(16, 1)
    # borrowed from https://www.manning.com/books/deep-learning-with-pytorch, Part 2, Chapter 11
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)
    # borrowed section ends here

    def forward(self, input_block, age_data):
        init_bn_out = self.initBatchNorm(input_block)
        block_out = self.block1(init_bn_out)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        # Flatten the output to shape Batch size x total size of block output
        linear_output = block_out.view(block_out.size(0), -1, )
        # add age value to penultimate Fully Connected layer if flag is set true
        if CONFIG.FEED_AGE:
            linear_output = torch.cat((linear_output, age_data), dim=1)
        linear_output = self.fc(linear_output)
        linear_output = self.fc2(linear_output)
        print(f'linear_output shape: {linear_output.shape},{linear_output}')

        return linear_output


class ConvBlock(nn.Module):
    def __init__(self, in_channels, conv_out_channels, max_need):
        super().__init__()
        self.convBatchNorm = nn.BatchNorm3d(conv_out_channels)

        self.conv1 = nn.Conv3d(in_channels, conv_out_channels, kernel_size=3, padding=0, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(conv_out_channels, conv_out_channels, kernel_size=3, padding=0, bias=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(2, 2)
        self.maxpool_toggle = max_need

    def forward(self, input_block):
        layer_out = self.conv1(input_block)
        layer_out = self.convBatchNorm(layer_out)
        layer_out = self.relu1(layer_out)
        layer_out = F.dropout(layer_out, p=0.35)
        layer_out = self.conv2(layer_out)
        layer_out = self.convBatchNorm(layer_out)
        layer_out = self.relu2(layer_out)
        layer_out = F.dropout(layer_out, p=0.35)
        if self.maxpool_toggle:
            return self.maxpool(layer_out)
        else:
            return layer_out
