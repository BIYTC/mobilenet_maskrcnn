# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
            self,
            proposal_matcher,
            fg_bg_sampler,
            box_coder,
            cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)  # target和proposal都是xywh格式的吗
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        # proposal.add_field("matched_idxs", matched_idxs)  # 后加的
        return matched_targets

    def prepare_targets(self, proposals, targets, first_iter=True, all_matched_targets=None):
        labels = []
        regression_targets = []
        if all_matched_targets is None:
            all_matched_targets = []
        if first_iter:
            for proposals_per_image, targets_per_image in zip(proposals, targets):
                # TODO:尝试只第一遍获取matched_idxs，后面不获取了
                # 但是这样做是错的，IOU的递进没有起作用
                matched_targets = self.match_targets_to_proposals(
                    proposals_per_image, targets_per_image
                )
                all_matched_targets.append(matched_targets)  # 后加的！！

                matched_idxs = matched_targets.get_field("matched_idxs")

                labels_per_image = matched_targets.get_field("labels")
                labels_per_image = labels_per_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD  # -1
                labels_per_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS  # -2
                labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

                # compute regression targets
                regression_targets_per_image = self.box_coder.encode(
                    matched_targets.bbox, proposals_per_image.bbox
                )

                labels.append(labels_per_image)
                regression_targets.append(regression_targets_per_image)
        else:
            for proposals_per_image, targets_per_image, matched_targets in zip(proposals, targets, all_matched_targets):
                # TODO:尝试只第一遍获取matched_idxs，后面不获取了
                # matched_targets = self.match_targets_to_proposals(
                #     proposals_per_image, targets_per_image
                # )
                # all_matched_targets.append(matched_targets)  # 后加的！！


                # matched_idxs = matched_targets.get_field("matched_idxs")

                labels_per_image = matched_targets.get_field("labels")
                labels_per_image = labels_per_image.to(dtype=torch.int64)

                matched_idxs = matched_targets.get_field("matched_idxs")
                bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_inds] = 0

                ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

                # compute regression targets
                regression_targets_per_image = self.box_coder.encode(
                    matched_targets.bbox, proposals_per_image.bbox
                )

                labels.append(labels_per_image)
                regression_targets.append(regression_targets_per_image)

        return labels, regression_targets, all_matched_targets

    def subsample(self, proposals, targets, first_iter=True, all_matched_targets=None):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, regression_targets, all_matched_targets = self.prepare_targets(proposals, targets,
                                                                               first_iter, all_matched_targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)  # 我不想让他做平衡!!!

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
                labels, regression_targets, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

            matched_targets_per_image = all_matched_targets[img_idx][img_sampled_inds]
            all_matched_targets[img_idx] = matched_targets_per_image

        self._proposals = proposals  # 这个函数把proposals加入了self
        return proposals, all_matched_targets

    def __call__(self, class_logits, box_regression, final_iter=True):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)  # 7类得分
        box_regression = cat(box_regression, dim=0)  # 每一个proposal打算要努力的方向
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)  # 实际属于第几类
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )  # 每一个proposal实际要努力的方向

        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)  # 不是背景的检测框都在什么位置
        labels_pos = labels[sampled_pos_inds_subset]  # 这些不是背景的检测框都属于第几类
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)  # 如果种类不明就都属于第一类
        elif final_iter:
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)  # 找到属于这个lable的box_regression对应的位置
        else:
            map_inds = torch.tensor([0, 1, 2, 3], device=device)  # 找到属于这个lable的box_regression对应的位置
        # note:
        # a:tensor([-1.3039,  0.0075,  0.0796, -0.3488, -0.0306])
        # b=a[:,None]
        # b:tensor([[-1.3039],
        #           [ 0.0075],
        #           [ 0.0796],
        #           [-0.3488],
        #           [-0.0306]])
        # d=b+torch.tensor([0.,1.,2.,3.])
        # d:tensor([[-1.3039, -0.3039,  0.6961,  1.6961],
        #           [ 0.0075,  1.0075,  2.0075,  3.0075],
        #           [ 0.0796,  1.0796,  2.0796,  3.0796],
        #           [-0.3488,  0.6512,  1.6512,  2.6512],
        #           [-0.0306,  0.9694,  1.9694,  2.9694]])
        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss


def make_roi_box_cascade_loss_evaluator(cfg, IOU, WEIGHTS):
    matcher = Matcher(
        IOU,
        0.3,
        allow_low_quality_matches=True,
    )

    bbox_reg_weights = WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )  # 512，0.25

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
