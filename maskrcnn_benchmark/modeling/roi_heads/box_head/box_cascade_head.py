# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_cascade_predictors import make_roi_box_cascade_predictor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss_cascade import make_roi_box_cascade_loss_evaluator


class ROIBoxCascadeHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxCascadeHead, self).__init__()
        self.cfg = cfg
        # self.feature_extractor = nn.ModuleList(
        #     [make_roi_box_feature_extractor(cfg, in_channels) for i in range(len(cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS))])
        self.feature_extractor_1 = make_roi_box_feature_extractor(cfg, in_channels)
        self.feature_extractor_2 = make_roi_box_feature_extractor(cfg, in_channels)
        self.feature_extractor_3 = make_roi_box_feature_extractor(cfg, in_channels)

        self.predictor_cascade_1 = make_roi_box_cascade_predictor(cfg, self.feature_extractor_1.out_channels)
        self.predictor_cascade_2 = make_roi_box_cascade_predictor(cfg, self.feature_extractor_2.out_channels)
        self.predictor = make_roi_box_predictor(cfg, self.feature_extractor_2.out_channels)
        # 返回各类得分与回归坐标

        self.post_processor = make_roi_box_post_processor(cfg)  # 输出结构后做NMS处理

        self.loss_evaluator_1 = make_roi_box_cascade_loss_evaluator(cfg, cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS[0],
                                                                    cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS[0])
        self.loss_evaluator_2 = make_roi_box_cascade_loss_evaluator(cfg, cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS[1],
                                                                    cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS[1])
        self.loss_evaluator_3 = make_roi_box_cascade_loss_evaluator(cfg, cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS[2],
                                                                    cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS[2])

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        loss_cls = []
        loss_reg = []
        proposals_new = proposals
        all_matched_targets = None
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                # 有了这一步才能算出loss
                proposals_new, all_matched_targets = self.loss_evaluator_1.subsample(proposals_new, targets)
        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads


        x = self.feature_extractor_1(features, proposals_new)
        x_ori = x
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor_cascade_1(x)
        # class_logits, box_regression = self.predictor(x)
        # class_logits_list.append(class_logits)
        if self.training:
            loss_classifier, loss_box_reg = self.loss_evaluator_1(
                [class_logits], [box_regression], final_iter=False
            )
            loss_cls.append(loss_classifier)
            loss_reg.append(loss_box_reg)
        result = self.post_processor(
            (class_logits, box_regression), proposals_new, final_iter=False)
        proposals_new = result

        # ------------------------------------------------------------------------------------------#
        # 2nd
        if self.training:
            with torch.no_grad():
                proposals_new, all_matched_targets = self.loss_evaluator_2.subsample(proposals_new, targets)
                # proposals_new, _ = self.loss_evaluator_2.subsample(proposals_new, targets,
                #                                                    first_iter=False,
                #                                                    all_matched_targets=all_matched_targets)
        x = self.feature_extractor_2(features, proposals_new)
        class_logits, box_regression = self.predictor_cascade_2(x)
        if self.training:
            loss_classifier, loss_box_reg = self.loss_evaluator_2([class_logits], [box_regression], final_iter=False)
            loss_cls.append(loss_classifier)
            loss_reg.append(loss_box_reg)
        result = self.post_processor(
            (class_logits, box_regression), proposals_new, final_iter=False)
        proposals_new = result
        # -------------------------------------------------------------------------------------------#
        # 3rd
        if self.training:
            with torch.no_grad():
                proposals_new, all_matched_targets = self.loss_evaluator_3.subsample(proposals_new, targets)
                # proposals_new, _ = self.loss_evaluator_3.subsample(proposals_new, targets,
                #                                                    first_iter=False,
                #                                                    all_matched_targets=all_matched_targets)
        x = self.feature_extractor_3(features, proposals_new)
        class_logits, box_regression = self.predictor(x)
        if self.training:
            loss_classifier, loss_box_reg = self.loss_evaluator_3([class_logits], [box_regression])
            loss_cls.append(loss_classifier)
            loss_reg.append(loss_box_reg)
        # result = self.post_processor(
        #     (class_logits, box_regression), proposals_new, final_iter=True)
        # -------------------------------------------------------------------------------------------#
        if not self.training:
            result = self.post_processor(
                (class_logits, box_regression), proposals_new, final_iter=True)
            return x_ori, result, {}

        loss_classifier = (loss_cls[0] + loss_cls[1] + loss_cls[2]) / 3
        loss_box_reg = (loss_reg[0] + loss_reg[1] + loss_reg[2]) / 3
        return (
            x_ori,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_cascade_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxCascadeHead(cfg, in_channels)
