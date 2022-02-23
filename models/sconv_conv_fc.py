'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = [
    'Sconv_2layer', 'Sconv_4layer', 'Sconv_6layer', 'Sconv_8layer', 's_conv_2layer', 's_conv_4layer', 's_conv_6layer', 's_conv_8layer',
]

######## nopool
class Sconv_2layer(nn.Module):
    '''
    Sconv model
    '''
    def __init__(self, features, n_channel, nunits, nclasses):
        super(Sconv_2layer, self).__init__()
        self.conv1 = nn.Conv2d(n_channel, nunits, kernel_size=9, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(nunits, nunits, kernel_size=13, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(nunits, nclasses)
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out


class Sconv_4layer(nn.Module):
    '''
    Sconv model
    '''
    def __init__(self, features, n_channel, nunits, nclasses):
        super(Sconv_4layer, self).__init__()
        self.conv1 = nn.Conv2d(n_channel, nunits, kernel_size=9, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(nunits, nunits, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(nunits, nunits, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(nunits, nunits, kernel_size=13, stride=1, padding=0)
        self.relu4 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(nunits, nclasses)
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out


class Sconv_6layer(nn.Module):
    '''
    Sconv model
    '''
    def __init__(self, features, n_channel, nunits, nclasses):
        super(Sconv_6layer, self).__init__()
        self.conv1 = nn.Conv2d(n_channel, nunits, kernel_size=9, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(nunits, nunits, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(nunits, nunits, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(nunits, nunits, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(nunits, nunits, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(nunits, nunits, kernel_size=13, stride=1, padding=0)
        self.relu6 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(nunits, nclasses)
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.conv5(out)
        out = self.relu5(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out


class Sconv_8layer(nn.Module):
    '''
    Sconv model
    '''
    def __init__(self, features, n_channel, nunits, nclasses):
        super(Sconv_8layer, self).__init__()
        self.conv1 = nn.Conv2d(n_channel, nunits, kernel_size=9, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(nunits, nunits, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(nunits, nunits, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(nunits, nunits, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(nunits, nunits, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(nunits, nunits, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(nunits, nunits, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(nunits, nunits, kernel_size=13, stride=1, padding=0)
        self.relu8 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(nunits, nclasses)
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.conv5(out)
        out = self.relu5(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out




def s_conv_2layer(nchannel, nunits, nclasses):
    """Sconv (configuration "A")"""
    return Sconv_2layer((), nchannel, nunits, nclasses)


def s_conv_4layer(nchannel, nunits, nclasses):
    """Sconv (configuration "A")"""
    return Sconv_4layer((), nchannel, nunits, nclasses)


def s_conv_6layer(nchannel, nunits, nclasses):
    """Sconv (configuration "A")"""
    return Sconv_6layer((), nchannel, nunits, nclasses)


def s_conv_8layer(nchannel, nunits, nclasses):
    """Sconv (configuration "A")"""
    return Sconv_8layer((), nchannel, nunits, nclasses)





