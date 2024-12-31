# Obtained from https://github.com/google-research/semivl

import torch
import torch.nn as nn
from torch.nn import functional as F

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


def ASPPConv(in_channels,
             out_channels,
             atrous_rate,
             norm=nn.BatchNorm2d,
             act=nn.ReLU):
    block = nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=atrous_rate,
            dilation=atrous_rate,
            bias=False), norm(out_channels), act(True))
    return block


class ASPPPooling(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm=nn.BatchNorm2d,
                 act=nn.ReLU):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm(out_channels), act(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):

    def __init__(self,
                 in_channels,
                 atrous_rates,
                 out_channels=None,
                 bn=True,
                 relu=True):
        super(ASPPModule, self).__init__()
        if out_channels is None:
            out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates
        norm = nn.BatchNorm2d if bn else nn.Identity
        act = nn.ReLU if relu else nn.Identity

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm(out_channels), act(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm, act)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm, act)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm, act)
        self.b4 = ASPPPooling(in_channels, out_channels, norm, act)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm(out_channels), act(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)


@MODELS.register_module()
class DLV3PHead(BaseDecodeHead):

    def __init__(self, c1_in_channels, c1_channels, dilations, img_size,
                 **kwargs):
        super(DLV3PHead, self).__init__(**kwargs)
        self.image_size = img_size
        self.aspp = ASPPModule(self.in_channels, dilations)
        self.c1_proj = nn.Sequential(
            nn.Conv2d(c1_in_channels, c1_channels, 1, bias=False),
            nn.BatchNorm2d(c1_channels), nn.ReLU(True))
        fuse_channels = self.in_channels // 8 + c1_channels
        self.head = nn.Sequential(
            nn.Conv2d(fuse_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256),
            nn.ReLU(True), nn.Conv2d(256, self.num_classes, 1, bias=True))
        self.conv_seg = None

    def forward(self, inputs, force_output_pred_masks=False):
        force_output_pred_masks = True
        if force_output_pred_masks:
            inputs = inputs[0][0]
        assert len(inputs) == 2
        c1, c4 = inputs[0], inputs[1]

        c4 = self.aspp(c4)
        c1 = self.c1_proj(c1)
        c4 = F.interpolate(
            c4,
            size=c1.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners)
        x = torch.cat([c1, c4], dim=1)
        out = self.head(x)

        if force_output_pred_masks:
            out = F.interpolate(
                out,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=self.align_corners)
            out = {"pred_masks": out}

        return out['pred_masks']
