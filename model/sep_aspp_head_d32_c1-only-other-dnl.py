import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, NonLocal2d

from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPHead, ASPPModule



class DisentangledNonLocal2d(NonLocal2d):
    """Disentangled Non-Local Blocks.

    Args:
        temperature (float): Temperature to adjust attention. Default: 0.05
    """

    def __init__(self, *arg, temperature=0.05, **kwargs):
        super().__init__(*arg, **kwargs)
        self.temperature = temperature
        self.conv_mask = nn.Conv2d(self.in_channels, 1, kernel_size=1)

    def embedded_gaussian(self, theta_x, phi_x):
        """Embedded gaussian with temperature."""

        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight /= self.temperature
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def forward(self, x):
        # x: [N, C, H, W]
        n = x.size(0)

        # g_x: [N, HxW, C]
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        if self.mode == 'gaussian':
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)
            else:
                phi_x = x.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)

        # subtract mean
        theta_x -= theta_x.mean(dim=-2, keepdim=True)
        phi_x -= phi_x.mean(dim=-1, keepdim=True)

        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *x.size()[2:])

        # unary_mask: [N, 1, HxW]
        unary_mask = self.conv_mask(x)
        unary_mask = unary_mask.view(n, 1, -1)
        unary_mask = unary_mask.softmax(dim=-1)
        # unary_x: [N, 1, C]
        unary_x = torch.matmul(unary_mask, g_x)
        # unary_x: [N, C, 1, 1]
        unary_x = unary_x.permute(0, 2, 1).contiguous().reshape(
            n, self.inter_channels, 1, 1)

        output = x + self.conv_out(y + unary_x)

        return output


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

    def __init__(self, c1_in_channels, c1_channels, reduction=2, use_scale=True, mode='embedded_gaussian', temperature=0.05,  **kwargs):
        super(DepthwiseSeparableASPPHead, self).__init__(**kwargs)
        assert c1_in_channels >= 0
        self.c2_channels = 512
        self.c3_channels = 1024
        self.c4_channels = 2048
        self.reduction = reduction
        self.use_scale = use_scale
        self.mode = mode
        self.temperature = temperature
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # if c1_in_channels > 0:
        #     self.c1_Dnl = DisentangledNonLocal2d(
        #         in_channels=c1_channels,
        #         reduction=self.reduction,
        #         use_scale=self.use_scale,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=self.norm_cfg,
        #         mode=self.mode,
        #         temperature=self.temperature)
        # else:
        #     self.c1_Dnl = None
        self.c2_Dnl = DisentangledNonLocal2d(
            in_channels=self.c2_channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode=self.mode,
            temperature=self.temperature)

        self.c3_Dnl = DisentangledNonLocal2d(
            in_channels=self.c3_channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode=self.mode,
            temperature=self.temperature)

        self.c4_Dnl = DisentangledNonLocal2d(
            in_channels=self.c4_channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode=self.mode,
            temperature=self.temperature)

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
        c4_output = self.c4_Dnl(inputs[3])
        # print("c4_output: ".format(c4_output.shape))
        output = torch.cat([output, c4_output], dim=1)      # channels = 2048+512
        output = self.bottleneck4(output)       # 3x3conv channels = 512
        # print("bottleneck4: {}".format(output.shape))
        # c3跨层 2倍上采样
        c3_output = self.c3_Dnl(inputs[2])
        output = resize(
            input=output,
            size=inputs[2].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output = torch.cat([output, c3_output], dim=1)
        output = self.bottleneck3(output)       # 3x3conv channels = 512
        # print("bottleneck3: {}".format(output.shape))
        # c2跨层 2倍上采样
        c2_output = self.c2_Dnl(inputs[1])
        output = resize(
            input=output,
            size=inputs[1].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output = torch.cat([output, c2_output], dim=1)
        output = self.bottleneck2(output)       # 3x3conv channels = 512
        # print("bottleneck2: {}".format(output.shape))

        # c1跨层 2倍上采样
        output = resize(
            input=output,
            size=inputs[0].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output = torch.cat([output, inputs[0]], dim=1)

        # if self.c1_Dnl is not None:
        #     c1_output = self.c1_Dnl(inputs[0])
        #     output = resize(
        #         input=output,
        #         size=inputs[0].shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        #     output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        # print("sep_bottleneck: {}".format(output.shape))
        output = self.cls_seg(output)
        # print("cls_seg: {}".format(output.shape))
        return output
