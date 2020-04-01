# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet, mobilenet
from torchsummary import summary
from . import bottom2up as panet_module


@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))  # 将主干加入网络
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)  # resnet网络结构
    summary(body, input_size=(3, 200, 200), device="cpu")
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS  # 256
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS  # 256 * 4
    # FPN结构
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU  # GN是否使用组归一化
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),  # 指定最后一层的输出是否需要再经过池化等操作，这里是最大值池化
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))  # 将body，fpn结构加入网络
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("MobileV2_FPN")
def build_resnet_fpn_backbone(cfg):
    body = mobilenet.MobileNetV2(cfg)  # resnet网络结构
    # in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS  # 256
    # out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS  # 256 * 4
    in_channels_stage2 = 64
    out_channels = 512
    # FPN结构
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU  # GN是否使用组归一化
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),  # 指定最后一层的输出是否需要再经过池化等操作，这里是最大值池化
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))  # 将body，fpn结构加入网络
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("MobileV2_PANET")
def build_resnet_fpn_backbone(cfg):
    body = mobilenet.MobileNetV2(cfg)  # resnet网络结构
    # in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS  # 256
    # out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS  # 256 * 4
    in_channels_stage2 = 64
    out_channels = 512
    # FPN结构
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU  # GN是否使用组归一化
        ),
        top_blocks=None,
    )
    # PANet结构
    bottom2up = panet_module.Bottom2UP(
        cfg,
        num_backbone_stages=3,
        in_channels=out_channels,
        top_blocks=panet_module.LastLevelMaxPool(),  # 指定最后一层的输出是否需要再经过池化等操作，这里是最大值池化
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn), ("bottom2up", bottom2up)]))  # 将body，fpn结构加入网络
    model.out_channels = out_channels
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
