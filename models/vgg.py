'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, nunits, nclasses):
        super(VGG, self).__init__()
        self.features = features
        mul = int(nunits / 16)
        fc_w = mul * 128
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(fc_w, fc_w),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(fc_w, fc_w),
            nn.ReLU(True),
            nn.Linear(fc_w, nclasses),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, nchannel, nunits, batch_norm=False):
    layers = []
    in_channels = nchannel
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            mul = int(nunits / 16)
            v = v * mul
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'B': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'D': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M'],
    'E': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 64, 'M', 128, 128, 128, 128, 'M',
          128, 128, 128, 128, 'M'],
}
# cfg = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
#           512, 512, 512, 512, 'M'],
# }



def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn(nchannel,nunits,nclasses):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], nchannel, nunits, batch_norm=True), nunits, nclasses)


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn(nchannel,nunits,nclasses):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], nchannel, nunits, batch_norm=True), nunits, nclasses)


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn(nchannel,nunits,nclasses):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], nchannel, nunits, batch_norm=True), nunits, nclasses)


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn(nchannel,nunits,nclasses):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], nchannel, nunits, batch_norm=True), nunits, nclasses)