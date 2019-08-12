#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : trident_resnet.py
#   Author      : chr
#   Created date: 2019-07-23 20:14:30
#   Description :
#
#================================================================

import logging
import torch.nn as nn
import torch

from ..registry import BACKBONES
from mmcv.runner import load_checkpoint
from mmcv.cnn import (constant_init, kaiming_init,
                      normal_init)


class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilation,
                 downsample=None):
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.downsample = downsample

        self.conv1 = nn.Conv2d(
                         self.in_channels,
                         self.out_channels // 4,
                         kernel_size=1,
                         stride=1,
                         bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
                         self.out_channels // 4,
                         self.out_channels // 4,
                         kernel_size=3,
                         stride=self.stride,
                         padding=1,
                         dilation=self.dilation,
                         bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
                         self.out_channels // 4,
                         self.out_channels,
                         kernel_size=1,
                         stride=1,
                         dilation=1,
                         bias=False)
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, data):
        identity = data
        print("Data:{}".format(data.size()))
        print("in_channels:{}".format(self.in_channels))
        print("out_channels:{}".format(self.out_channels))
        out = self.conv1(data)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(data)

        out += identity
        out = self.relu3(out)

        return out


class TridentBlock(nn.Module):

    def __init__(self,
                in_channels,
                out_channels,
                stride,
                dilation,
                downsample=None):
        super(TridentBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.downsample = downsample

        self.conv1 = nn.Conv2d(self.in_channels,
                               self.out_channels // 4,
                               kernel_size=1,
                               stride=1,
                               dilation=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(self.out_channels // 4,
                               self.out_channels // 4,
                               kernel_size=3,
                               padding=self.dilation,
                               stride=self.stride,
                               dilation=self.dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(self.out_channels // 4,
                               self.out_channels,
                               kernel_size=1,
                               stride=1,
                               dilation=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, data):
        identity = data
        print("TridentStage_0.In function dataType:{}".format(type(data)))

        out = self.conv1(data)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(data)

        print("Trident_Stage out:{}".format(out.size()))
        print("Trident_Stage Data:{}".format(identity.size()))
        print("Trident_Stage Dilation:{}".format(self.dilation))

        out += identity
        out = self.relu3(out)

        return out


class resnet_C1(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, use_3x3_conv0):

        super(resnet_C1, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_3x3_conv0 = use_3x3_conv0
        self.layers = list()

        if isinstance(self.kernel_size, (list, tuple)):
            self.padding = [int(ii / 2) for ii in self.kernel_size]
        else:
            self.padding = int(self.kernel_size / 2)

        assert isinstance(use_3x3_conv0, bool)

        if self.use_3x3_conv0:

            self.layers.append(nn.Conv2d(
                             self.in_channels,
                             self.out_channels,
                             kernel_size=3,
                             stride=self.stride,
                             padding=self.padding,
                             dilation=self.dilation,
                             bias=False))
            self.layers.append(nn.BatchNorm2d(self.out_channels))
            self.layers.append(nn.ReLU(inplace=True))

            self.layers.append(nn.Conv2d(
                             self.out_channels,
                             self.out_channels,
                             kernel_size=3,
                             stride=1,
                             padding=self.padding,
                             dilation=self.dilation,
                             bias=False))
            self.layers.append(nn.BatchNorm2d( self.out_channels))
            self.layers.append(nn.ReLU(inplace=True))

            self.layers.append(nn.Conv2d(
                             self.out_channels,
                             self.out_channels,
                             kernel_size=3,
                             stride=1,
                             padding=self.padding,
                             dilation=self.dilation,
                             bias=False))
            self.layers.append(nn.BatchNorm2d(self.out_channels))
            self.layers.append(nn.ReLU(inplace=True))
        else:
            self.layers.append(nn.Conv2d(
                             self.in_channels,
                             self.out_channels,
                             kernel_size=self.kernel_size,
                             stride=self.stride,
                             padding=self.padding,
                             dilation=self.dilation,
                             bias=False))
            self.layers.append(nn.BatchNorm2d(self.out_channels))
            self.layers.append(nn.ReLU(inplace=True))

        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layers = nn.Sequential(*self.layers)

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, /' \
            ' stride={stride}, padding={padding}, use_3x3_conv0={use_3x3_conv0})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, data):
        data = self.layers(data)
        print("The output feature map of ResNet_C1:{}".format(data.size()))
        return data


def make_layer(block, blocks, in_channels, out_channels, stride, dilation):

    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=1,
                      stride=stride,
                      dilation=1,
                      bias=False),
            nn.BatchNorm2d(out_channels)
        )
    layers = list()
    layers.append(block(in_channels, out_channels, stride, dilation, downsample))
    print("Original in_channels:{}".format(in_channels))
    in_channels = out_channels
    print("Changed in_channels:{}".format(in_channels))

    for _ in range(1, blocks):
        layers.append(block(in_channels, out_channels, 1, dilation))

    return nn.Sequential(*layers)


class resnet_stage(nn.Module):

    def __init__(self, in_channels, out_channels, stride, dilation, block, blocks):
        super(resnet_stage, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.block = block
        self.blocks = blocks

        self.layers = make_layer(self.block,
                                 self.blocks,
                                 self.in_channels,
                                 self.out_channels,
                                 self.stride,
                                 self.dilation)

    def forward(self, data):
        data = self.layers(data)
        return data


class resnet_trident_stage(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 block,
                 trident_block,
                 blocks,
                 branch_dilates,
                 num_trident_block,
                 num_branch,
                 branch_ids,
                 branch_bn_shared,
                 branch_conv_shared):

        super(resnet_trident_stage, self).__init__()
        assert isinstance(branch_dilates, list) \
               and len(branch_dilates) == len(branch_ids) == num_branch, \
            'dilate should be a list with num_branch items.'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.block = block
        self.trident_block = trident_block
        self.blocks = blocks
        self.branch_dilates = branch_dilates
        self.num_trident_block = num_trident_block
        self.num_branch = num_branch
        self.branch_ids = branch_ids
        self.branch_bn_shared = branch_bn_shared
        self.branch_conv_shared = branch_conv_shared

        self.layer1 = make_layer(self.block, self.blocks - self.num_trident_block, self.in_channels,
                                 self.out_channels, self.stride, 1)
        self.layer2 = [] * 3
        for i in range(0, self.num_branch):
            self.layer2.append(make_layer(
                                        self.trident_block,
                                        self.num_trident_block,
                                        self.out_channels,
                                        self.out_channels,
                                        self.stride,
                                        self.branch_dilates[i]))

    def forward(self, data):
        data = self.layer1(data)
        print("TridentStage_0.data:{}".format(data.size()))
        print("TridentStage_0.dataType:{}".format(type(data)))
        data = data.cpu()
        print("TridentStage_0.Change dataType:{}".format(type(data)))
        data_0 = self.layer2[0](data)
        data_1 = self.layer2[1](data)
        data_2 = self.layer2[2](data)

        data = [data_0, data_1, data_2]

        return data


@BACKBONES.register_module
class TridentResNet(nn.Module):
    depth_config = {
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3),
        200: (3, 24, 36, 3)
    }

    def __init__(self, depth, num_stages=4, stride=(1, 2, 2, 2), style='pytorch'):
        super(TridentResNet, self).__init__()
        self.depth = depth
        assert self.depth in [50, 101, 152, 200]
        self.stage_blocks = self.depth_config[self.depth]
        self.num_stages = num_stages
        assert self.num_stages >= 1 and self.num_stages <= 4

        self.layer1 = resnet_C1(3, 64, 3, 2, 1, True)
        self.layer2 = resnet_stage(64, 256, stride=1, dilation=1, block=Bottleneck, blocks=self.stage_blocks[0])
        self.layer3 = resnet_stage(256, 512, stride=2, dilation=1, block=Bottleneck, blocks=self.stage_blocks[1])
        self.layer4 = resnet_trident_stage(512, 1024,
                                           stride=2,
                                           block=Bottleneck,
                                           trident_block=TridentBlock,
                                           blocks=self.stage_blocks[2],
                                           branch_dilates=[1, 2, 3],
                                           num_trident_block=3,
                                           num_branch=3,
                                           branch_ids=[0, 1, 2],
                                           branch_conv_shared=True,
                                           branch_bn_shared=True)
        # self.layer5 = resnet_stage(512, 1024, stride=2, dilation=1, block=Bottleneck, blocks=self.stage_blocks[3])

    def forward(self, data):
        data = self.layer1(data)
        data = self.layer2(data)
        data = self.layer3(data)
        data = self.layer4(data)

        return data

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)

        else:
            raise TypeError("pretrained must be a str or None")









