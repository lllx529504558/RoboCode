import torch
import torch.nn as nn

# MobileNet
def conv_dw(in_planes, out_planes, stride):
    return nn.Sequential(
        nn.Conv2d(in_planes, in_planes, 3, stride, 1, groups=in_planes, bias=False),
        nn.BatchNorm2d(in_planes),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

class MobileNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(MobileNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.conv1 = conv_dw(32, 64, 1)
        self.conv2 = conv_dw(64, 128, 2)
        self.conv3 = conv_dw(128, 128, 1)
        self.conv4 = conv_dw(128, 256, 2)
        self.conv5 = conv_dw(256, 256, 1)
        self.conv6 = conv_dw(256, 512, 2)
        self.conv7 = conv_dw(512, 512, 1)
        self.conv8 = conv_dw(512, 512, 1)
        self.conv9 = conv_dw(512, 512, 1)
        self.conv10 = conv_dw(512, 512, 1)
        self.conv11 = conv_dw(512, 1024, 2)
        self.conv12 = conv_dw(1024, 1024, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        out0 = self.pre(x)
        out1 = self.conv1(out0)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)
        out7 = self.conv7(out6)
        out8 = self.conv8(out7)
        out9 = self.conv9(out8)
        out10 = self.conv10(out9)
        out11 = self.conv11(out10)
        out12 = self.conv12(out11)
        return {'out': out12, 'low_level': out1}