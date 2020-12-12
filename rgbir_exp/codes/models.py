"""
Angular Triplet Loss
YE, Hanrong et al, Bi-directional Exponential Angular Triplet Loss for RGB-Infrared Person Re-Identification
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from backbone import ResNet
import settings


class FeatureEmbedder(nn.Module):
    def __init__(self, in_planes=2048):
        super(FeatureEmbedder, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x):
        global_feat = self.gap(x)  # (b, 2048, 1, 1)
        feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        bnfeat = self.bottleneck(feat)  # normalize for angular softmax
        return bnfeat

class IdClassifier(nn.Module):
    def __init__(self, in_planes = 2048, num_classes = settings.num_classes): # train 296, val 99
        super(IdClassifier, self).__init__()
        self.classifier = nn.Linear(in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    def __init__(self, last_stride, model_path):
        super(Baseline, self).__init__()
        self.base = ResNet(last_stride)
        self.base.load_param(model_path)

    def forward(self, x):
        return  self.base(x)  # (b, 2048, 1, 1)

