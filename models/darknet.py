import torch
from torch import nn
from torch.nn import functional as F

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
    """BasicBlock"""
    expansion = 2
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
            
        self.downsample = downsample
        self.stride = stride
        
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.01)
        self.conv2 = conv3x3(planes, inplanes, stride=stride)
        self.bn2 = nn.BatchNorm2d(inplanes, momentum=0.01)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        
        if self.downsample is not None:
            x = self.downsample(x)

        fx = self.conv1(x)
        fx = self.bn1(fx)
        fx = self.lrelu(fx)

        fx = self.conv2(fx)
        fx = self.bn2(fx)
        fx = self.lrelu(fx)
        
        return fx + x

class DarkNet(nn.Module):
    """DarkNet"""
    def __init__(self, layers=[1, 2, 8, 8, 4], num_classes=1000, block=BasicBlock):
        super().__init__()
        self.layers = layers
        self.inplanes=32
        # input
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, momentum=0.01)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        # base
        self.layer1 = self._make_layer(block,  32, layers[0], stride=1)
        self.layer2 = self._make_layer(block,  64, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=1)
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
                conv3x3(self.inplanes, planes * block.expansion, stride=2),
                nn.BatchNorm2d(planes * block.expansion),
                nn.LeakyReLU(0.1, inplace=True))

        layers = []
        self.inplanes = planes * block.expansion
        layers.append(block(self.inplanes, planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
def darknet19(num_classes=1000, **kwards):
    """darknet21"""
    return DarkNet([1, 1, 2, 2, 1], num_classes, BasicBlock)
    
def darknet53(num_classes=1000, **kwards):
    """darknet53"""
    return DarkNet([1, 2, 8, 8, 4], num_classes, BasicBlock)