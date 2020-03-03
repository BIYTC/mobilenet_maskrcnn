# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
            self, in_channels_list, out_channels, conv_block, top_blocks=None
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):  # 起始索引为1
            inner_block = "fpn_inner{}".format(idx)  # 用下表起名: fpn_inner1, fpn_inner2, fpn_inner3, fpn_inner4
            layer_block = "fpn_layer{}".format(idx)  # 用下表起名: fpn_layer1, fpn_layer2, fpn_layer3, fpn_layer4

            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1)  # 该1*1卷积层主要作用为改变通道数为out_channels
            layer_block_module = conv_block(out_channels, out_channels, 3,
                                            1)  # 用3*3卷积对融合结果卷积，消除上采样的混叠效(aliasing effect)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks  # 将top_blocks作为FPN类成员变量，指定最后一层的输出是否需要再经过池化等操作，这里是最大值池化

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.输入分辨率最高的位于第一个，先取最后特征图最小的来运算，
            然后倒序用for循环做后面的运算。
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.   分辨率最高的位于第一
        """
        # getattr() 函数用于返回一个对象属性值
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])  # 先计算最后一层（分辨率最低）特征图的fpn结果
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        '''
        [::-1]列表逆置，为了让分辨率小的特征图位于前面，例：
        x=[0,1,2,3,4,5,6,7,8,9]
        x[:-1][::-1]
        [8, 7, 6, 5, 4, 3, 2, 1, 0]
        '''
        for feature, inner_block, layer_block in zip(  # 上面做了一次，这里应该再重复做三次
                x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")  # 进行2倍最近临上采样
            inner_lateral = getattr(self, inner_block)(feature)  # 做1*1卷积
            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
            # mode='bilinear', align_corners=False)
            last_inner = inner_lateral + inner_top_down  # 求和
            results.insert(0, getattr(self, layer_block)(last_inner))  # 对last_inner做3*3卷积。为了使分辨率最大的在前，将结果插入到0位置

        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)  # results不是一个特征图，是4张


class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]
