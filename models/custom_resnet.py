'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

#PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
# Layer1 -
# X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]

# R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
# Add(X, R1)

# Layer 2 -
# Conv 3x3 [256k]
# MaxPooling2D
# BN
# ReLU

# Layer 3 -
# X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]

# R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
# Add(X, R2)

# MaxPooling with Kernel Size 4

# FC Layer 
# SoftMax
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64


        #PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Layer1 -
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(2,2)
        self.bn2 = nn.BatchNorm2d(128)


        # R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)


        # Layer 2 -
        # Conv 3x3 [256k]
        # MaxPooling2D
        # BN
        # ReLU
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.pool3 = nn.MaxPool2d(2,2)
        self.bn3 = nn.BatchNorm2d(256)

        # Layer 3 -
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.pool4 = nn.MaxPool2d(2,2)
        self.bn4 = nn.BatchNorm2d(512)

        # R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        # Add(X, R2)
        self.layer2 = self._make_layer(block, 512, num_blocks[1], stride=2)

        self.pool5 = nn.MaxPool2d(4,4)

        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = F.relu(self.bn2(self.pool2(self.conv2(out))))


        out = self.layer1(out)

        out = F.relu(self.bn3(self.pool3(self.conv3(out))))

        out = F.relu(self.bn4(self.pool4(self.conv4(out))))

        out = self.layer2(out)

        # out = self.layer3(out)
        # out = self.layer4(out)
        out = self.pool5(out)

        out = out.view(out.size(0), -1)
        out = self.linear(out)

        out = F.softmax(out)
        return out


def Custom_ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
