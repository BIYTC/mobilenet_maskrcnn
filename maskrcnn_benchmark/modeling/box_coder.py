# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    这个类将一组边界框编码和解码为用于训练回归器的表示
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip  # 边框的长和宽的最高值

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """

        TO_REMOVE = 1  # TODO remove
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE  # 计算候选框的宽度
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE  # 计算候选框的高度
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths  # 计算候选框中心的ｘ坐标
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights  # 计算候选框中心的ｙ坐标

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE  # 计算基准边框（ground truth)的宽度
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE  # 计算基准边框（ground truth)的高度
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths  # 计算基准边框（ground truth)中心的ｘ坐标
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights  # 计算基准边框（ground truth)中心的ｙ坐标

        wx, wy, ww, wh = self.weights
        # 计算带有权重的回归目标的各个部分
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        # 将回归目标的各个部分合并为一个元组，并依次保存到一个栈里
        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.
        根据得到的候选框  以及  与之对应的中心X,Y,W,H的各部分的回归值和得到预测边框

        Arguments:
            rel_codes (Tensor): encoded boxes
            根据候选框与基准边框（ground truth)的差距计算出来的候选边框中心X,Y,W,H的各部分的 变差回归值
            boxes (Tensor): reference boxes.候选边框
        """

        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

        return pred_boxes
