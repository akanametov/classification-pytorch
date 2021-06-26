import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['DenseNet', 'densenet23', 'densenet37', 'densenet69', 'densenet93',
           'densenet121', 'densenet169', 'densenet201', 'densenet264']

def conv3x3(inplanes, outplanes, **kwargs):
    """3x3 convolution with padding"""
    kwargs['kernel_size'] = 3
    kwargs['padding'] = 1
    kwargs['bias'] = False
    return nn.Conv2d(inplanes, outplanes, **kwargs)

def conv1x1(inplanes, outplanes, **kwargs):
    """1x1 convolution"""
    kwargs['kernel_size'] = 1
    kwargs['bias'] = False
    return nn.Conv2d(inplanes, outplanes, **kwargs)

class BasicBlock(nn.Module):
    expansion = 2
    def __init__(self, inplanes, growth_rate=32, drop_rate=0):
        super(BasicBlock, self).__init__()
        planes = growth_rate * self.expansion
        self.drop_rate = drop_rate
        
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv1x1(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, growth_rate)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        
        fx = self.bn1(x)
        fx = self.relu(fx)
        fx = self.conv1(fx)
        fx = self.bn2(fx)
        fx = self.relu(fx)
        fx = self.conv2(fx)
        
        if self.drop_rate > 0:
            fx = F.dropout(fx, p=self.drop_rate, training=self.training)

        fx = torch.cat([x, fx], 1)

        return fx

class DenseBlock(nn.Module):
    expansion = 4
    def __init__(self, inplanes, growth_rate=32, drop_rate=0):
        super(DenseBlock, self).__init__()
        planes = growth_rate * self.expansion
        self.drop_rate = drop_rate
        
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv1x1(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, growth_rate)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        
        fx = self.bn1(x)
        fx = self.relu(fx)
        fx = self.conv1(fx)
        fx = self.bn2(fx)
        fx = self.relu(fx)
        fx = self.conv2(fx)
        
        if self.drop_rate > 0:
            fx = F.dropout(fx, p=self.drop_rate, training=self.training)

        fx = torch.cat([x, fx], 1)

        return fx


class TransitionLayer(nn.Module):
    compression = 2
    def __init__(self, inplanes):
        super(TransitionLayer, self).__init__()
        planes = inplanes//self.compression
        
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(inplanes, planes)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        

    def forward(self, x):
        
        fx = self.bn1(x)
        fx = self.relu(fx)
        fx = self.conv1(fx)
        fx = self.avgpool(fx)
        
        return fx


class DenseNet(nn.Module):
    """DenseNet"""
    def __init__(self, layers, num_classes=1000,
                 growth_rate=32, drop_rate=0,
                 block=DenseBlock, layer=TransitionLayer):
        super(DenseNet, self).__init__()
        self.layers=layers
        self.num_classes=num_classes
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        self.inplanes=64
        # input
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # base
        self.block1 = self._make_block(block, layers[0])
        self.layer1 = self._make_layer(layer)
        self.block2 = self._make_block(block, layers[1])
        self.layer2 = self._make_layer(layer)
        self.block3 = self._make_block(block, layers[2])
        self.layer3 = self._make_layer(layer)
        self.block4 = self._make_block(block, layers[3])
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #output
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplanes, num_classes)
        # init weights
        self.initialize_weights()
        
    def initialize_weights(self,):
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return None
        
    def _make_block(self, block, blocks):
        layers = []
        for i in range(blocks):
            layers.append(block(self.inplanes, growth_rate=self.growth_rate, drop_rate=self.drop_rate))
            self.inplanes += self.growth_rate
        return nn.Sequential(*layers)

    def _make_layer(self, layer):
        inplanes = self.inplanes
        transition = layer(inplanes)
        self.inplanes = self.inplanes//layer.compression
        return transition


    def forward(self, x):
        x = self.conv1(x)
        
        x = self.block1(x)
        x = self.layer1(x) 
        x = self.block2(x)
        x = self.layer2(x) 
        x = self.block3(x)
        x = self.layer3(x) 
        x = self.block4(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
def densenet23(num_classes=1000, **kwargs):
    """densenet23"""
    return DenseNet([2, 2, 3, 2], num_classes=num_classes, block=BasicBlock, **kwargs)
    
def densenet37(num_classes=1000, **kwargs):
    """densenet37"""
    return DenseNet([2, 4, 6, 4], num_classes=num_classes, block=BasicBlock, **kwargs)

def densenet69(num_classes=1000, **kwargs):
    """densenet69"""
    return DenseNet([4, 6, 14, 8], num_classes=num_classes, block=BasicBlock, **kwargs)

def densenet93(num_classes=1000, **kwargs):
    """densenet93"""
    return DenseNet([6, 8, 18, 12], num_classes=num_classes, block=BasicBlock, **kwargs)

def densenet121(num_classes=1000, **kwargs):
    """densenet121"""
    return DenseNet([6, 12, 24, 16], num_classes, **kwargs)

def densenet169(num_classes=1000, **kwargs):
    """densenet169"""
    return DenseNet([6, 12, 32, 32], num_classes, **kwargs)

def densenet201(num_classes=1000, **kwargs):
    """densenet201"""
    return DenseNet([6, 12, 48, 32], num_classes, **kwargs)

def densenet264(num_classes=1000, **kwargs):
    """densenet264"""
    return DenseNet([6, 12, 64, 48], num_classes, **kwargs)