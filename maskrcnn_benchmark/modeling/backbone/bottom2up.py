import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.modeling.make_layers import group_norm


class Bottom2UP(nn.Module):
    """
    Module that adds PANet on a list of feature maps from FPN.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
            self, cfg, in_channels, num_backbone_stages, top_blocks,
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
        super(Bottom2UP, self).__init__()
        # self.inner_blocks = []
        # self.layer_blocks = []
        # for idx, in_channels in enumerate(in_channels_list, 1):  # 起始索引为1
        #     inner_block = "fpn_inner{}".format(idx)  # 用下表起名: fpn_inner1, fpn_inner2, fpn_inner3, fpn_inner4
        #     layer_block = "fpn_layer{}".format(idx)  # 用下表起名: fpn_layer1, fpn_layer2, fpn_layer3, fpn_layer4
        #
        #     if in_channels == 0:
        #         continue
        #     inner_block_module = conv_block(in_channels, out_channels, 1)  # 该1*1卷积层主要作用为改变通道数为out_channels
        #     layer_block_module = conv_block(out_channels, out_channels, 3,
        #                                     1)  # 用3*3卷积对融合结果卷积，消除上采样的混叠效(aliasing effect)
        #     self.add_module(inner_block, inner_block_module)
        #     self.add_module(layer_block, layer_block_module)
        #     self.inner_blocks.append(inner_block)
        #     self.layer_blocks.append(layer_block)
        # self.top_blocks = top_blocks  # 将top_blocks作为FPN类成员变量，指定最后一层的输出是否需要再经过池化等操作，这里是最大值池化
        self.panet_buttomup_conv1_modules = nn.ModuleList()
        self.panet_buttomup_conv2_modules = nn.ModuleList()
        for i in range(num_backbone_stages):
            if cfg.MODEL.FPN.PANET.USE_GN:
                self.panet_buttomup_conv1_modules.append(nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=True),  # 下采样
                    group_norm(in_channels),
                    nn.ReLU(inplace=True)
                ))
                self.panet_buttomup_conv2_modules.append(nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),  # 像素相加后使用
                    group_norm(in_channels),
                    nn.ReLU(inplace=True)
                ))
            else:
                self.panet_buttomup_conv1_modules.append(
                    nn.Conv2d(in_channels, in_channels, 3, 2, 1)
                )
                self.panet_buttomup_conv2_modules.append(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1)
                )
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.输入分辨率最高的位于第一个，先取第一个特征图最大的来运算，
            然后用for循环做后面的运算。
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.   分辨率最高的位于第一
        """
        # getattr() 函数用于返回一个对象属性值
        last_feature = x[0]  # 第一层不做处理
        results = []
        results.append(last_feature)

        for feature, buttomup_conv1, buttomup_conv2 in zip(
                x[1:], self.panet_buttomup_conv1_modules, self.panet_buttomup_conv2_modules
        ):
            inner_feature = buttomup_conv1(last_feature)
            last_feature = feature + inner_feature
            last_feature = buttomup_conv2(last_feature)
            results.append(last_feature)

        if isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])  # 这还加入了一张特征图,对最小的特征图进行了池化
            results.extend(last_results)

        return tuple(results)  # results不是一个特征图，是5张


class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]
