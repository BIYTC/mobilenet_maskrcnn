# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.layers import ROIAlign

from .utils import cat


class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
        LevelMapper函数的作用是获得某个特征区域将会从网络的那一层特征上进行提取，面积越大的目标区往往会在高层进行提取，小目标则在低层卷基层
    上进行特征提取。本函数的主要目标就是确定某个目标最好从那一层上进行提取
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        """
        Arguments:
            boxlists (list[BoxList])
        """
        # Compute level ids
        s = torch.sqrt(cat([boxlist.area() for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min


class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scales, sampling_ratio):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
                )
            )
        self.poolers = nn.ModuleList(poolers)  # 加入模型
        self.output_size = output_size
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(boxes)
        if num_levels == 1:
            return self.poolers[0](x[0], rois)

        levels = self.map_levels(boxes)

        num_rois = len(rois)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros(
            (num_rois, num_channels, output_size, output_size),
            dtype=dtype,
            device=device,
        )
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):  # 获得所有应该从同一特征层提取特征的roi
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            result[idx_in_level] = pooler(per_level_feature, rois_per_level).to(dtype) # 将大小相似的这些目标特征送入到特定同一个特征层进行池化，得到相应的结果

        return result


def make_pooler(cfg, head_name):
    resolution = cfg.MODEL[head_name].POOLER_RESOLUTION
    scales = cfg.MODEL[head_name].POOLER_SCALES
    sampling_ratio = cfg.MODEL[head_name].POOLER_SAMPLING_RATIO
    pooler = Pooler(
        output_size=(resolution, resolution),
        scales=scales,
        sampling_ratio=sampling_ratio,
    )
    return pooler
