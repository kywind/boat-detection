import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import resnet18


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = resnet18(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x):
        x   = self.conv1(x) 
        x   = self.bn1(x)
        x   = self.relu(x)     # 1/2, 64
        x   = self.maxpool(x)  # 1/4, 64
        f4  = self.layer1(x)   # 1/4, 64
        f8  = self.layer2(f4)   # 1/8, 128
        f16 = self.layer3(f8)   # 1/16, 256

        return f16, f8, f4


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv2d(128, 64, 1)
        self.conv2 = nn.Conv2d(64, 32, 1)
        self.conv3 = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        f16, f8, f4 = self.encoder(x)

        x = F.interpolate(f8, scale_factor=2, mode='bilinear', align_corners=False) 
        x = self.conv1(x)
        x = F.relu(x)
        x = x + f4

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) 
        x = self.conv2(x)
        x = F.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) 
        x = self.conv3(x) 
        
        return x.squeeze(1)


if __name__ == '__main__':
    Net()