from __future__ import division
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResNeXt', 'resnext50', 'resnext101', 'resnext152']


def conv3x3(in_planes, out_planes, **kwargs):
    """3x3 convolution with padding"""
    kwargs['kernel_size'] = 3
    kwargs['padding'] = 1
    kwargs['bias'] = False
    return nn.Conv2d(in_planes, out_planes, **kwargs)

def conv1x1(in_planes, out_planes, **kwargs):
    """1x1 convolution"""
    kwargs['kernel_size'] = 1
    kwargs['bias'] = False
    return nn.Conv2d(in_planes, out_planes, **kwargs)

class Bottleneck(nn.Module):
    """Bottleneck"""
    expansion = 4

    def __init__(self, inplanes, planes, base_width, cardinality, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        mid_planes = cardinality*int(math.floor(planes*(base_width/64)))

        self.conv1 = conv1x1(inplanes, mid_planes)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = conv3x3(mid_planes, mid_planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = conv1x1(mid_planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        fx = x

        fx = self.conv1(fx)
        fx = self.bn1(fx)
        fx = self.relu(fx)

        fx = self.conv2(fx)
        fx = self.bn2(fx)
        fx = self.relu(fx)

        fx = self.conv3(fx)
        fx = self.bn3(fx)

        if self.downsample is not None:
            x = self.downsample(x)

        fx = fx + x
        fx = self.relu(fx)

        return fx


class ResNeXt(nn.Module):
    """ResNeXt"""
    def __init__(self, layers, num_classes,
                 cardinality=32, base_width=4, block=Bottleneck):
        super(ResNeXt, self).__init__()
        self.layers=layers
        self.num_classes=num_classes
        self.base_width = base_width
        self.cardinality = cardinality
        self.inplanes = 64
        # input
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # base
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # output
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # init weights
        self.initialize_weights()
        
    def initialize_weights(self,):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return None
        

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, self.base_width, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.base_width, self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnext50(num_classes=1000, **kwargs):
    """resnet50"""
    return ResNeXt([3, 4, 6, 3], num_classes, **kwargs)


def resnext101(num_classes=1000, **kwargs):
    """resnet101"""
    return ResNeXt([3, 4, 23, 3], num_classes, **kwargs)


def resnext152(num_classes=1000, **kwargs):
    """resnet152"""
    return ResNeXt([3, 8, 36, 3], num_classes, **kwargs)