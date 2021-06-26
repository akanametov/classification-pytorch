import torch
import torch.nn as nn

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

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

class BasicBlock(nn.Module):
    """BasicBlock"""
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        fx = x

        fx = self.conv1(fx)
        fx = self.bn1(fx)
        fx = self.relu(fx)

        fx = self.conv2(fx)
        fx = self.bn2(fx)

        if self.downsample is not None:
            x = self.downsample(x)

        fx = fx + x
        fx = self.relu(fx)

        return fx
    
class Bottleneck(nn.Module):
    """Bottleneck"""
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
    
class ResNet(nn.Module):
    """ResNet"""
    def __init__(self, layers, num_classes=1000, block=BasicBlock):
        super(ResNet, self).__init__()
        self.layers=layers
        self.num_classes=num_classes
        self.inplanes=64
        # input
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return None

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
def resnet18(num_classes=1000, **kwargs):
    """resnet18"""
    return ResNet([2, 2, 2, 2], num_classes, BasicBlock)

def resnet34(num_classes=1000, **kwargs):
    """resnet34"""
    return ResNet([3, 4, 6, 3], num_classes, BasicBlock)

def resnet50(num_classes=1000, **kwargs):
    """resnet50"""
    return ResNet([3, 4, 6, 3], num_classes, Bottleneck)

def resnet101(num_classes=1000, **kwargs):
    """resnet101"""
    return ResNet([3, 4, 23, 3], num_classes, Bottleneck)

def resnet152(num_classes=1000, **kwargs):
    """resnet152"""
    return ResNet([3, 8, 36, 3], num_classes, Bottleneck)