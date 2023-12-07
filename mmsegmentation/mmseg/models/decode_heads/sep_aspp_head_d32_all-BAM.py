import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPHead, ASPPModule


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        mid_channel = channel // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )

    def forward(self, x):
        avg = self.avg_pool(x).view(x.size(0), -1)
        out = self.shared_MLP(avg).unsqueeze(2).unsqueeze(3).expand_as(x)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, channel, reduction=16, dilation_rate=4):
        super(SpatialAttention, self).__init__()
        mid_channel = channel // reduction
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(channel, mid_channel, kernel_size=1),
            nn.SyncBatchNorm(mid_channel),
            nn.ReLU(inplace=True)
        )
        self.dilations_convs = nn.Sequential(
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=dilation_rate, dilation=dilation_rate),
            nn.SyncBatchNorm(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=dilation_rate, dilation=dilation_rate),
            nn.SyncBatchNorm(mid_channel),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(mid_channel, channel, kernel_size=1)

    def forward(self, x):
        y = self.reduce_conv(x)
        x = self.dilations_convs(y)
        out = self.final_conv(x)
        return out



class BAM(nn.Module):
    """
        BAM: Bottleneck Attention Module
        https://arxiv.org/pdf/1807.06514.pdf
    """
    def __init__(self, channel):
        super(BAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention(channel)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        att = 1 + self.sigmoid(self.channel_attention(x) * self.spatial_attention(x))
        return att * x


class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)


@HEADS.register_module()
class DepthwiseSeparableASPPHead(ASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, c1_in_channels, c1_channels, **kwargs):
        super(DepthwiseSeparableASPPHead, self).__init__(**kwargs)
        assert c1_in_channels >= 0
        self.c2_channels = 512
        self.c3_channels = 1024
        self.c4_channels = 2048
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if c1_in_channels > 0:
            self.c1_BAM = BAM(c1_in_channels)
        else:
            self.c1_BAM = None
        self.c2_BAM = BAM(self.c2_channels)
        self.c3_BAM = BAM(self.c3_channels)
        self.c4_BAM = BAM(self.c4_channels)

        self.bottleneck2 = ConvModule(
            self.channels + self.c2_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck3 = ConvModule(
            self.channels + self.c3_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck4 = ConvModule(
            self.channels + self.c4_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)     # 3x3conv channels = 512
        # print("aspp_outs bottleneck: {}".format(output.shape))
        # c4跨层
        c4_output = self.c4_BAM(inputs[3])
        # print("c4_output: ".format(c4_output.shape))
        output = torch.cat([output, c4_output], dim=1)      # channels = 2048+512
        output = self.bottleneck4(output)       # 3x3conv channels = 512
        # print("bottleneck4: {}".format(output.shape))
        # c3跨层 2倍上采样
        c3_output = self.c3_BAM(inputs[2])
        output = resize(
            input=output,
            size=inputs[2].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output = torch.cat([output, c3_output], dim=1)
        output = self.bottleneck3(output)       # 3x3conv channels = 512
        # print("bottleneck3: {}".format(output.shape))
        # c2跨层 2倍上采样
        c2_output = self.c2_BAM(inputs[1])
        output = resize(
            input=output,
            size=inputs[1].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output = torch.cat([output, c2_output], dim=1)
        output = self.bottleneck2(output)       # 3x3conv channels = 512
        # print("bottleneck2: {}".format(output.shape))

        # c1跨层 2倍上采样
        # output = resize(
        #     input=output,
        #     size=inputs[0].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # output = torch.cat([output, inputs[0]], dim=1)

        if self.c1_BAM is not None:
            c1_output = self.c1_BAM(inputs[0])
            output = resize(
                input=output,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        # print("sep_bottleneck: {}".format(output.shape))
        output = self.cls_seg(output)
        # print("cls_seg: {}".format(output.shape))
        return output
