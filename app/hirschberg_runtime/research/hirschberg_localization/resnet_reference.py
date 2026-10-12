"""Frozen inference subset imported from the training workspace."""
import torch
from torch import nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inc, out, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inc, out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + identity)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inc, out, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inc, out, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.conv3 = nn.Conv2d(out, out * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.relu(self.bn3(self.conv3(x)) + identity)

class ResNet(nn.Module):

    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._layer(block, 64, layers[0])
        self.layer2 = self._layer(block, 128, layers[1], 2)
        self.layer3 = self._layer(block, 256, layers[2], 2)
        self.layer4 = self._layer(block, 512, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _layer(self, block, out, count, stride=1):
        down = None
        if stride != 1 or self.inplanes != out * block.expansion:
            down = nn.Sequential(nn.Conv2d(self.inplanes, out * block.expansion, 1, stride, bias=False), nn.BatchNorm2d(out * block.expansion))
        layers = [block(self.inplanes, out, stride, down)]
        self.inplanes = out * block.expansion
        layers.extend((block(self.inplanes, out) for _ in range(1, count)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
        return self.fc(torch.flatten(self.avgpool(x), 1))

def resnet(name):
    if name == 'resnet18':
        return ResNet(BasicBlock, [2, 2, 2, 2])
    if name == 'resnet50':
        return ResNet(Bottleneck, [3, 4, 6, 3])
    raise ValueError(name)
