#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

# This file is modified from the open_lth repository: https://github.com/facebookresearch/open_lth
import torch.nn as nn
import torch.nn.functional as F


class MLP1_layer(nn.Module):
    """A 1-hidden-layer MLP network for CIFAR-10."""

    # def conv1x1(
    #         self,
    #         in_planes,
    #         out_planes,
    #         stride=1,
    #         groups=1,
    #         first_layer=False,
    #         last_layer=False,
    #         is_conv=False,
    # ):
    #     """1x1 convolution with padding"""
    #     c = self.conv(
    #         1,
    #         in_planes,
    #         out_planes,
    #         stride=stride,
    #         groups=groups,
    #         first_layer=first_layer,
    #         last_layer=last_layer,
    #         is_conv=is_conv,
    #     )
    #
    #     return c

    def __init__(self):
        super(MLP1_layer, self).__init__()

        # self.fc1 = conv1x1(1, 32 * 32 * 3, 1024)
        # self.fc2 = conv1x1(1, 1024, 10)

        self.fc1 = nn.Conv1d(32 * 32 * 3, 1024, 1)
        self.drop1= nn.Dropout(p=0.5)
        self.fc2 = nn.Conv1d(1024, 10, 1)


    def forward(self, x):
        '''Forward pass'''
        ### with dropout
        out1 = F.relu(self.drop1(self.fc1(x)))
        # out1 = F.relu(self.fc1(x))
        out2 = self.fc2(out1)

        return out2


