# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import numpy as np
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(
            self,
            sizes=(128, 256, 512),
            aspect_ratios=(0.5, 1.0, 2.0),
            anchor_strides=(8, 16, 32),
            straddle_thresh=0,
    ):
        super(AnchorGenerator, self).__init__()

        if len(anchor_strides) == 1:
            anchor_stride = anchor_strides[0]
            cell_anchors = [
                generate_anchors(anchor_stride, sizes, aspect_ratios).float()
            ]  # 不用FPN的话stride只有一种，对应多个size
        else:  # 再次确认使用FPN，校验是否正确
            if len(anchor_strides) != len(sizes):
                raise RuntimeError("FPN should have #anchor_strides == #sizes")

            cell_anchors = [
                generate_anchors(
                    anchor_stride,
                    size if isinstance(size, (tuple, list)) else (size,),
                    aspect_ratios
                ).float()
                for anchor_stride, size in zip(anchor_strides, sizes)  # anchor_stride和size是一一对应的,
                # 一个stride对应一种size
            ]
        self.strides = anchor_strides
        self.cell_anchors = BufferList(cell_anchors)
        self.straddle_thresh = straddle_thresh

    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):  # 按照步长从图像的*左上角*向全图滑动，从坐上向下，右，按照步长划遍全图，从全图截取anchor
        # grid_sizes为在某个特征层上特征图的大小，要求stride，gride size和FPN输出层的个数需要是一致的
        anchors = []
        for size, stride, base_anchors in zip(
                grid_sizes, self.strides, self.cell_anchors
        ):
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(
                0, grid_width * stride, step=stride, dtype=torch.float32, device=device  # 生成anchor在x轴上的滑动距离的列表
            )
            shifts_y = torch.arange(
                0, grid_height * stride, step=stride, dtype=torch.float32, device=device  # 生成anchor在y轴上的滑动距离的列表
            )
            # 由anchor在x轴和y轴上需要滑动的距离列表构造出滑动距离的网格，即要滑动完全图anchor在x,y平面的滑动向量
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)  # 把anchor在x,y轴上的滑动距离的列表压缩成一维数组
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)  # 把anchor在x,y轴上的滑动距离的数组构造出二维的滑动网格

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
                # 计算得到anchor在各个滑动节点上的目标，包括很多超出图片范围的anchor

            )

        return anchors

    def add_visibility_to(self, boxlist):  # 判断是否越界，即是否visibility
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
        if self.straddle_thresh >= 0:
            inds_inside = (
                (anchors[..., 0] >= -self.straddle_thresh)
                & (anchors[..., 1] >= -self.straddle_thresh)
                & (anchors[..., 2] < image_width + self.straddle_thresh)
                & (anchors[..., 3] < image_height + self.straddle_thresh)
            )
        else:
            device = anchors.device
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.bool, device=device)
        boxlist.add_field("visibility", inds_inside)

    def forward(self, image_list, feature_maps):
        """
        生成anchor
        :param image_list: 传入模型的图片数据，数据维度为：n_pic * width * height
        :param feature_maps: 每个图片经过backbone之后提取的各个特征图,　数据维度为：　n_layer * n_pic * map_width * map_height
        :return: anchors：　返回生成的anchor
        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors = []
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                boxlist = BoxList(
                    anchors_per_feature_map, (image_width, image_height), mode="xyxy"
                )
                self.add_visibility_to(boxlist)
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors


def make_anchor_generator(config):  # anchor的大小，可以为多个，其大小是针对源图像的
    anchor_sizes = config.MODEL.RPN.ANCHOR_SIZES  # (4, 8, 16, 32, 64)
    aspect_ratios = config.MODEL.RPN.ASPECT_RATIOS  # (0.1, 1.0/5.0, 1.0/3.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
    anchor_stride = config.MODEL.RPN.ANCHOR_STRIDE  # (4, 8, 16, 32, 64)
    straddle_thresh = config.MODEL.RPN.STRADDLE_THRESH  # 0

    if config.MODEL.RPN.USE_FPN:  # FPN需要保证，stride个数，size个数以及FPN的层数相同。
        assert len(anchor_stride) == len(
            anchor_sizes
        ), "FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)"
    else:
        assert len(anchor_stride) == 1, "Non-FPN should have a single ANCHOR_STRIDE"
    anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios, anchor_stride, straddle_thresh  # straddle_thresh可以标记anchor是否超出边界，
        # 界内的visibility为1，越界的为0
    )
    return anchor_generator


def make_anchor_generator_retinanet(config):
    anchor_sizes = config.MODEL.RETINANET.ANCHOR_SIZES
    aspect_ratios = config.MODEL.RETINANET.ASPECT_RATIOS
    anchor_strides = config.MODEL.RETINANET.ANCHOR_STRIDES
    straddle_thresh = config.MODEL.RETINANET.STRADDLE_THRESH
    octave = config.MODEL.RETINANET.OCTAVE
    scales_per_octave = config.MODEL.RETINANET.SCALES_PER_OCTAVE

    assert len(anchor_strides) == len(anchor_sizes), "Only support FPN now"
    new_anchor_sizes = []
    for size in anchor_sizes:
        per_layer_anchor_sizes = []
        for scale_per_octave in range(scales_per_octave):
            octave_scale = octave ** (scale_per_octave / float(scales_per_octave))
            per_layer_anchor_sizes.append(octave_scale * size)
        new_anchor_sizes.append(tuple(per_layer_anchor_sizes))

    anchor_generator = AnchorGenerator(
        tuple(new_anchor_sizes), aspect_ratios, anchor_strides, straddle_thresh
    )
    return anchor_generator


# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#        [-175.,  -87.,  192.,  104.],
#        [-359., -183.,  376.,  200.],
#        [ -55.,  -55.,   72.,   72.],
#        [-119., -119.,  136.,  136.],
#        [-247., -247.,  264.,  264.],
#        [ -35.,  -79.,   52.,   96.],
#        [ -79., -167.,   96.,  184.],
#        [-167., -343.,  184.,  360.]])


def generate_anchors(
        stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors(
        stride,
        np.array(sizes, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float),
    )


def _generate_anchors(base_size, scales, aspect_ratios):  # base_size对应步长stride; scales对应特征图上的正方形锚框边长，
    # 在相同的stride下是由size决定的
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size],
                      dtype=np.float) - 1  # 特征图上的一个点对应原图上一个(0, 0, base_size - 1, base_size - 1)窗口，
    # 分别代表这个窗口区域的左下角和右上角的点的坐标
    anchors = _ratio_enum(anchor, aspect_ratios)  # 特征图一个点生成的anchors
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]  # 生成的新anchor是按照size来的
    )
    return torch.from_numpy(anchors)


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    把保存anchor宽，高，中心点的坐标信息的数据结构转换为保存anchor左下角和右上角两个点的坐标的anchor数据类型
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )
    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)  # 返回的是原图锚点框长宽和中心点坐标(锚点框自己的相对坐标，不是在原图上的坐标)
    size = w * h  # 原图锚框面积
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor.
    把每个anchor都按照scales来扩充, scales为size/stride。先用stride构造基本的anchor再扩充scales倍即可
    原来一个点对应stride*stride，乘以scales后为（stride*size/stride)*（stride*size/stride)"""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
