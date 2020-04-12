# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F
import torch
from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.poolers import Pooler, AdaptivePooler
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3, group_norm

registry.ROI_MASK_FEATURE_EXTRACTORS.register(
    "ResNet50Conv5ROIFeatureExtractor", ResNet50Conv5ROIFeatureExtractor
)


@registry.ROI_MASK_FEATURE_EXTRACTORS.register("MaskRCNNFPNFeatureExtractor")
class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION  # 14
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES  # (0.25, 0.125, 0.0625, 0.03125)
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO  # 2
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn
            )  # 这里用到膨胀卷积了
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x


@registry.ROI_MASK_FEATURE_EXTRACTORS.register("MaskRCNNPANETFeatureExtractor")
class MaskRCNNPANETFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNPANETFeatureExtractor, self).__init__()
        self.cfg = cfg
        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION  # 14
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES  # (0.25, 0.125, 0.0625, 0.03125)
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO  # 2
        pooler = AdaptivePooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS  # (256, 256, 256, 256)
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION

        next_feature = input_size
        self.blocks = []
        # for layer_idx, layer_features in enumerate(layers, 1):
        #     layer_name = "mask_fcn{}".format(layer_idx)
        #     module = make_conv3x3(
        #         next_feature, layer_features,
        #         dilation=dilation, stride=1, use_gn=use_gn
        #     )  # 这里用到膨胀卷积了
        #     self.add_module(layer_name, module)
        #     next_feature = layer_features
        #     self.blocks.append(layer_name)
        self.add_module("mask_fcn1_1",
                        make_conv3x3(next_feature, layers[0], dilation=dilation, stride=1, use_gn=use_gn))
        self.add_module("mask_fcn1_2",
                        make_conv3x3(next_feature, layers[0], dilation=dilation, stride=1, use_gn=use_gn))
        self.add_module("mask_fcn1_3",
                        make_conv3x3(next_feature, layers[0], dilation=dilation, stride=1, use_gn=use_gn))
        self.add_module("mask_fcn1_4",
                        make_conv3x3(next_feature, layers[0], dilation=dilation, stride=1, use_gn=use_gn))
        next_feature = layers[0]
        for layer_idx, layer_features in enumerate(layers[1:], 2):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn
            )  # 这里用到膨胀卷积了
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        # TODO:区分前后景所需的模块，需要初始化权重！！！
        conv4 = nn.Conv2d(layers[2], layers[2], 3, 1, padding=1 * dilation, dilation=dilation, bias=False)
        nn.init.kaiming_normal_(
            conv4.weight, mode="fan_out", nonlinearity="relu"
        )
        self.mask_conv4_fc = nn.Sequential(
            conv4,
            group_norm(layers[2]),
            nn.ReLU(inplace=True))
        # --------------------------------------------------------------------------------------------------------#
        conv5 = nn.Conv2d(layers[2], int(layers[2] / 2), 3, 1, padding=1 * dilation, dilation=dilation, bias=False)
        nn.init.kaiming_normal_(
            conv5.weight, mode="fan_out", nonlinearity="relu"
        )
        self.mask_conv5_fc = nn.Sequential(
            conv5,
            group_norm(int(layers[2] / 2)),
            nn.ReLU(inplace=True))
        # self.mask_conv5_fc = nn.Sequential(
        #     nn.Conv2d(layers[2], int(layers[2] / 2), 3, 1, padding=1 * dilation, dilation=dilation, bias=False),
        #     group_norm(int(layers[2] / 2)),
        #     nn.ReLU(inplace=True))
        # nn.init.kaiming_normal_(
        #     self.mask_conv5_fc.weight, mode="fan_out", nonlinearity="relu"
        # )
        #---------------------------------------------------------------------------------------------------------#
        fc = nn.Linear(int(layers[2] / 2) * cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION ** 2,
                       (2 * cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION) ** 2, bias=True)
        nn.init.kaiming_normal_(
            fc.weight, mode="fan_out", nonlinearity="relu"
        )
        self.mask_fc = nn.Sequential(
            fc,
            nn.ReLU(inplace=True))
        # self.mask_fc = nn.Sequential(
        #     nn.Linear(int(layers[2] / 2) * cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION ** 2,
        #               (2 * cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION) ** 2, bias=True),
        #     nn.ReLU(inplace=True))
        # nn.init.kaiming_normal_(
        #     self.mask_fc.weight, mode="fan_out", nonlinearity="relu"
        # )

        self.out_channels = layer_features

    def forward(self, x, proposals):
        pooled_fearures = self.pooler(x, proposals)
        x_0 = self.mask_fcn1_1(pooled_fearures[:, 0, :, :, :])
        x_1 = self.mask_fcn1_2(pooled_fearures[:, 1, :, :, :])
        x_2 = self.mask_fcn1_3(pooled_fearures[:, 2, :, :, :])
        x_3 = self.mask_fcn1_4(pooled_fearures[:, 3, :, :, :])
        x = torch.stack([x_0, x_1, x_2, x_3], dim=1)
        x, _ = torch.max(x, 1)
        x = F.relu(self.mask_fcn2(x))
        x_fc3 = F.relu(self.mask_fcn3(x))
        x = F.relu(self.mask_fcn4(x_fc3))

        # TODO:构建区分前后景的分支
        x_ff = self.mask_conv4_fc(x_fc3)
        x_ff = self.mask_conv5_fc(x_ff)
        x_ff = self.mask_fc(x_ff.view(x_fc3.size(0), -1))
        x_ff = x_ff.view(-1, 1, 2 * self.cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION,
                         2 * self.cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION)
        x_ff = x_ff.repeat(1, self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES, 1, 1)
        return x, x_ff


def make_roi_mask_feature_extractor(cfg, in_channels):
    func = registry.ROI_MASK_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
