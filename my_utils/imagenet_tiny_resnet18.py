"""
ResNet18 for Tiny-ImageNet (200 classes)
Manual implementation (CIFAR-style: 3x3 conv1, no maxpool)
All BatchNorm replaced with GroupNorm for federated learning
"""

import torch
import torch.nn as nn
from flgo.utils import fmodule


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=2, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=2, num_channels=out_channels * BasicBlock.expansion),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=2, num_channels=out_channels * BasicBlock.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(fmodule.FModule):
    def __init__(self, num_classes=200):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=2, num_channels=64),
            nn.ReLU(inplace=True)
        )

        self.conv2_x = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.conv3_x = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.conv4_x = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.conv5_x = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def init_local_module(object):
    pass


def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = ResNet(num_classes=200).to(object.device)