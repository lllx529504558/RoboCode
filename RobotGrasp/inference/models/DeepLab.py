import torch
import torch.nn as nn
import torch.nn.functional as F
from inference.models.backbone.ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from inference.models.backbone.MobileNet import MobileNet

class ASPP(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, atrous_rates=[6, 12, 18], pool=32):
        super(ASPP, self).__init__()
        self.aspps = nn.ModuleList()
        self.aspps.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        for rate in atrous_rates:
            self.aspps.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.aspps.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((pool, pool)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (2 + len(atrous_rates)), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        out = [aspp(x) for aspp in self.aspps]
        out = torch.cat(out, dim=1)
        out = self.project(out)
        return out


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, num_classes, low_level_channels, aspp_dilate=[6, 12, 18]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.projrct = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.aspp = ASPP(in_channels=in_channels, out_channels=256, atrous_rates=aspp_dilate, pool=32)
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()
    
    def forward(self, feature):
        low_level_feature = self.projrct(feature['low_level']) # return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        out = self.aspp(feature['out'])
        out = F.interpolate(out, size=low_level_feature.size()[-2:], mode='bilinear', align_corners=False)
        out = torch.cat((out, low_level_feature), dim=1)
        out = self.classifier(out)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)     


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels, num_classes, low_level_channels=256, aspp_dilate=[6, 12, 18], backbone='ResNet50'):
        super(DeepLabV3Plus, self).__init__()
        if backbone == 'ResNet18':
            self.backbone = ResNet18(in_channels, num_classes)
            self.classifier = DeepLabHeadV3Plus(512, num_classes, 64, aspp_dilate)
        elif backbone == 'ResNet34':
            self.backbone = ResNet34(in_channels, num_classes)
            self.classifier = DeepLabHeadV3Plus(512, num_classes, 64, aspp_dilate)
        elif backbone == 'ResNet50':
            self.backbone = ResNet50(in_channels, num_classes)
            self.classifier = DeepLabHeadV3Plus(2048, num_classes, 256, aspp_dilate)
        elif backbone == 'ResNet101':
            self.backbone = ResNet101(in_channels, num_classes)
            self.classifier = DeepLabHeadV3Plus(2048, num_classes, 256, aspp_dilate)
        elif backbone == 'ResNet152':
            self.backbone = ResNet152(in_channels, num_classes)
            self.classifier = DeepLabHeadV3Plus(2048, num_classes, 256, aspp_dilate)
        elif backbone == 'MobileNet':
            self.backbone = MobileNet(in_channels, num_classes)
            self.classifier = DeepLabHeadV3Plus(1024, num_classes, 64, aspp_dilate, pool=16)
    def forward(self, x):
        out = self.backbone(x)
        out = self.classifier(out)
        return out

        