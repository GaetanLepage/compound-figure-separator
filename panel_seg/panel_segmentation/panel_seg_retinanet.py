"""
#############################
#        CompFigSep         #
# Compound Figure Separator #
#############################

GitHub:         https://github.com/GaetanLepage/compound-figure-separator

Author:         Gaétan Lepage
Email:          gaetan.lepage@grenoble-inp.org
Date:           Spring 2020

Master's project @HES-SO (Sierre, SW)

Supervisors:    Henning Müller (henning.mueller@hevs.ch)
                Manfredo Atzori (manfredo.atzori@hevs.ch)

Collaborator:   Niccolò Marini (niccolo.marini@hevs.ch)


##################################################################
Custom model derived from RetinaNet to achieve panel segmentation.
"""

import math
from typing import List, Tuple, Dict
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn

from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.config import CfgNode

from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone.resnet import build_resnet_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher

from detectron2.modeling.backbone.fpn import FPN, LastLevelP6P7
from detectron2.modeling.meta_arch.retinanet import (
    permute_to_N_HWA_K,
    permute_all_cls_and_box_to_N_HWA_K_and_concat
)

__all__ = ["PanelSegRetinaNet"]


def build_fpn_backbones(cfg: CfgNode,
                        input_shape: ShapeSpec) -> Tuple[nn.Module, nn.Module]:
    """
    Build the Resnet 50 backbone and the two FPN networks:
    * Panel FPN:
        * Input:  C3, C4, C5
        * Output: P3, P4, P5, P6, P7
    * Label FPN:
        * Input:  C2, C3, C4
        * Output: P2, P3, P4

    Args:
        cfg (CfgNode): a detectron2 CfgNode
        input_shape (ShapeSpec): TODO

    Returns:
        backbone (Backbone):    backbone module, must be a subclass of :class:`Backbone`.
    """
    # Build the feature extractor (Resnet 50)
    bottom_up = build_resnet_backbone(cfg, input_shape)

    # Panel FPN
    panel_in_features = cfg.MODEL.PANEL_FPN.IN_FEATURES
    panel_out_channels = cfg.MODEL.PANEL_FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()["res5"].channels
    panel_fpn = FPN(bottom_up=bottom_up,
                    in_features=panel_in_features,
                    out_channels=panel_out_channels,
                    norm=cfg.MODEL.FPN.NORM,
                    top_block=LastLevelP6P7(in_channels_p6p7, panel_out_channels),
                    fuse_type=cfg.MODEL.FPN.FUSE_TYPE)


    # Label FPN
    label_in_features = cfg.MODEL.LABEL_FPN.IN_FEATURES
    label_out_channels = cfg.MODEL.LABEL_FPN.OUT_CHANNELS
    label_fpn = FPN(bottom_up=bottom_up,
                    in_features=label_in_features,
                    out_channels=label_out_channels,
                    norm=cfg.MODEL.FPN.NORM,
                    top_block=None,
                    fuse_type=cfg.MODEL.FPN.FUSE_TYPE)

    return panel_fpn, label_fpn


class PanelSegRetinaNet(nn.Module):
    """
    Custom model derived from RetinaNet to achieve panel segmentation.

    This model has been introduced by Zou et al.
    (https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/asi.24334)

    Attributes:
        num_label_classes (int):                Number of label classes.
        panel_in_features (List[str]):          List of the names of the panel heads inputs.
                                                    ex: ['p3', 'p4', 'p5', 'p6', 'p7']
        label_in_features (List[str]):          List of the names of the label heads inputs.
                                                    ex: ['p2', 'p3', 'p4']
        focal_loss_alpha (float):               Alpha parameter for the focal loss.
        focal_loss_gamma (float):               Gamma parameter for the focal loss.
        smooth_l1_loss_beta (float):            Beta parameter for the L1 loss (regression loss).
        score_threshold (float):                Minimum score for a detection to be kept
                                                    (inference).
        topk_candidates (int):                  Maximum number of detections per image.
        nms_threshold (float):                  IoU threshold used in NMS filtering process
                                                    (inference).
        max_detections_per_image (int):         Maximum number of detections to return per image
                                                    during inference.
        panel_fpn (nn.Module):                  Panel Feature Pyramid Network (FPN).
        label_fpn (nn.Module):                  Label Feature Pyramid Network (FPN).
        panel_anchor_generator (DefaultAnchorGenerator):
                                                Compute anchors for panel detection.
        panel_head (RetinaNetHead):             RetinaNetHead handling panels prediction.
        label_anchor_generator (DefaultAnchorGenerator):
                                                Compute anchors for label detection.
        label_head (RetinaNet):                 RetinaNetHead handling panels prediction.
        box2box_transform (Box2BoxTransform):   Box2box transformation as defined in R-CNN.
        matcher (Matcher):                      Match predicted element to a ground-truth element.
        loss_normalizer (int):                  Reasonable number of foreground examples.
        loss_normalizer_momentum (float):       The loss normalizer momentum.
    """

    def __init__(self, cfg: CfgNode):
        super().__init__()

        self.num_label_classes = cfg.MODEL.RETINANET.NUM_LABEL_CLASSES
        self.panel_in_features = cfg.MODEL.RETINANET.PANEL_IN_FEATURES
        self.label_in_features = cfg.MODEL.RETINANET.LABEL_IN_FEATURES
        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta = cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA
        # Inference parameters:
        self.score_threshold = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.RETINANET.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE

        self.panel_fpn, self.label_fpn = build_fpn_backbones(
            cfg,
            input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        # Panel
        panel_backbone_fpn_output_shape = self.panel_fpn.output_shape()
        panel_feature_shapes = [panel_backbone_fpn_output_shape[f]
                                for f in self.panel_in_features]

        self.panel_anchor_generator = DefaultAnchorGenerator(
            sizes=cfg.MODEL.PANEL_ANCHOR_GENERATOR.SIZES,
            aspect_ratios=cfg.MODEL.PANEL_ANCHOR_GENERATOR.ASPECT_RATIOS,
            strides=[x.stride for x in panel_feature_shapes],
            offset=cfg.MODEL.ANCHOR_GENERATOR.OFFSET)

        self.panel_head = RetinaNetHead(cfg,
                                        input_shape=panel_feature_shapes,
                                        num_classes=1,
                                        num_anchors=self.panel_anchor_generator.num_cell_anchors)


        # Label
        label_backbone_fpn_output_shape = self.label_fpn.output_shape()
        label_feature_shapes = [label_backbone_fpn_output_shape[f]
                                for f in self.label_in_features]

        self.label_anchor_generator = DefaultAnchorGenerator(
            sizes=cfg.MODEL.LABEL_ANCHOR_GENERATOR.SIZES,
            aspect_ratios=cfg.MODEL.LABEL_ANCHOR_GENERATOR.ASPECT_RATIOS,
            strides=[x.stride for x in label_feature_shapes],
            offset=cfg.MODEL.ANCHOR_GENERATOR.OFFSET)

        self.label_head = RetinaNetHead(cfg,
                                        input_shape=label_feature_shapes,
                                        num_classes=cfg.MODEL.RETINANET.NUM_LABEL_CLASSES,
                                        num_anchors=self.label_anchor_generator.num_cell_anchors)

        # Matching and loss
        # TODO maybe have to duplicate those as well
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.matcher = Matcher(cfg.MODEL.RETINANET.IOU_THRESHOLDS,
                               cfg.MODEL.RETINANET.IOU_LABELS,
                               allow_low_quality_matches=True)

        self.register_buffer("pixel_mean",
                             torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std",
                             torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        """
        In Detectron1, loss is normalized by number of foreground samples in the batch.
        When batch size is 1 per GPU, #foreground has a large variance and
        using it lead to lower performance. Here we maintain an EMA of #foreground to
        stabilize the normalizer.
        """
        self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9


    @property
    def device(self) -> torch.device:
        """
        Identify the device the model is hosted by.

        Returns:
            device (torch.device):  The model device.
        """
        return self.pixel_mean.device


    def forward(self, batched_inputs: List[dict]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batched_inputs (List[dict]):
                A list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            Dict[str, Tensor]:  Mapping from a named loss to a scalar tensor storing the loss.
                                    Used during training only. The dict keys are:
                                    'panel_loss_cls', 'panel_loss_box_reg', 'label_loss_cls'
                                    and 'label_loss_box_reg'.
        """
        images = self.preprocess_image(batched_inputs)

        # detected panels
        panel_features = self.panel_fpn(images.tensor)
        panel_features = [panel_features[f] for f in self.panel_in_features]
        panel_anchors = self.panel_anchor_generator(panel_features)
        panel_cls, panel_box_delta = self.panel_head(panel_features)

        # detected labels
        label_features = self.label_fpn(images.tensor)
        label_features = [label_features[f] for f in self.label_in_features]
        label_anchors = self.label_anchor_generator(label_features)
        label_cls, label_box_delta = self.label_head(label_features)

        # Training
        if self.training:

            # Panels
            panel_gt_instances = [x['panel_instances'].to(self.device)
                                  for x in batched_inputs]
            panel_gt_classes, panel_gt_anchors_reg_deltas = self.get_ground_truth(
                panel_anchors,
                gt_instances=panel_gt_instances,
                num_classes=1)

            panel_loss_cls, panel_loss_box_reg = self._compute_single_head_losses(
                gt_classes=panel_gt_classes,
                gt_anchors_deltas=panel_gt_anchors_reg_deltas,
                pred_class_logits=panel_cls,
                pred_anchor_deltas=panel_box_delta,
                num_classes=1)

            loss_dict = {
                'panel_loss_cls': panel_loss_cls,
                'panel_loss_box_reg': panel_loss_box_reg
            }

            # Labels
            label_gt_instances = [x['label_instances'].to(self.device)
                                  for x in batched_inputs]

            label_gt_classes, label_gt_anchors_reg_deltas = self.get_ground_truth(
                label_anchors,
                gt_instances=label_gt_instances,
                num_classes=self.num_label_classes)
            label_loss_cls, label_loss_box_reg = self._compute_single_head_losses(
                gt_classes=label_gt_classes,
                gt_anchors_deltas=label_gt_anchors_reg_deltas,
                pred_class_logits=label_cls,
                pred_anchor_deltas=label_box_delta,
                num_classes=self.num_label_classes)

            loss_dict['label_loss_cls'] = label_loss_cls
            loss_dict['label_loss_box_reg'] = label_loss_box_reg

            return loss_dict

        # Otherwise, do inference.
        batched_inference_results = self.inference(panel_cls,
                                                   panel_box_delta,
                                                   panel_anchors,
                                                   label_cls,
                                                   label_box_delta,
                                                   label_anchors,
                                                   images.image_sizes)
        processed_results = []
        for inference_results, input_per_image, image_size in zip(
                batched_inference_results,
                batched_inputs,
                images.image_sizes):

            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            # TODO check that this work with two sets of boxes
            # r = detector_postprocess(results_per_image, height, width)

            panel_results, label_results = inference_results

            scale_x, scale_y = (width / panel_results.image_size[1],
                                height / panel_results.image_size[0])

            # 1) Panels
            panel_results = Instances((height, width), **panel_results.get_fields())

            # Clip and scale boxes
            panel_output_boxes = panel_results.pred_boxes
            panel_output_boxes.scale(scale_x, scale_y)
            panel_output_boxes.clip(panel_results.image_size)

            # 2) Labels
            label_results = Instances((height, width), **label_results.get_fields())

            # Clip and scale boxes
            label_output_boxes = label_results.pred_boxes
            label_output_boxes.scale(scale_x, scale_y)
            label_output_boxes.clip(label_results.image_size)

            processed_results.append({"panels": panel_results, "labels": label_results})

        return processed_results


    def _compute_single_head_losses(self,
                                    gt_classes: torch.Tensor,
                                    gt_anchors_deltas: torch.Tensor,
                                    pred_class_logits: torch.Tensor,
                                    pred_anchor_deltas: torch.Tensor,
                                    num_classes: int) -> Tuple[float, float]:
        """
        Compute the loss dict for a single RetinaNet branch (one classification head and one
        regression head).

        N is the number of images in the batch.
        R is the total number of anchors.
        The total number of anchors across levels, i.e. sum(Hi x Wi x A)

        Args:
            gt_classes (torch.Tensor):          Ground truth classes.
                                                    shape=(N, R).
            gt_anchors_deltas (torch.Tensor):   Ground truth box deltas.
                                                    shape=(N, R, 4).
            pred_class_logits (torch.Tensor):   Predicted classes for the batch images.
                                                    shape=(N, R).
            pred_anchor_deltas (torch.Tensor):  Predicted box deltas.
                                                    shape=(N, R, 4).

        Returns:
            loss_cls, loss_box_reg (Tuple[float, float]):   The value of the loss for each head
                                                                (classification and regression).
        """

        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            box_cls=pred_class_logits,
            box_delta=pred_anchor_deltas,
            num_classes=num_classes)

        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != num_classes)
        num_foreground = foreground_idxs.sum().item()
        get_event_storage().put_scalar("num_foreground", num_foreground)
        self.loss_normalizer = (
            self.loss_normalizer_momentum * self.loss_normalizer
            + (1 - self.loss_normalizer_momentum) * num_foreground
        )

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(pred_class_logits[valid_idxs],
                                          gt_classes_target[valid_idxs],
                                          alpha=self.focal_loss_alpha,
                                          gamma=self.focal_loss_gamma,
                                          reduction="sum") / max(1, self.loss_normalizer)

        # regression loss
        loss_box_reg = smooth_l1_loss(pred_anchor_deltas[foreground_idxs],
                                      gt_anchors_deltas[foreground_idxs],
                                      beta=self.smooth_l1_loss_beta,
                                      reduction="sum") / max(1, self.loss_normalizer)

        return loss_cls, loss_box_reg


    @torch.no_grad()
    def get_ground_truth(self,
                         anchors: List[Boxes],
                         gt_instances: List[Instances],
                         num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract the ground truth classes and deltas from a list of Instances objects.

        Args:
            anchors (list[Boxes]):      A list of #feature level Boxes. The Boxes contains
                                            anchors of this image on the specific feature level.
            targets (list[Instances]):  A list of N `Instances`s. The i-th `Instances` contains
                                            the ground-truth per-instance annotations for the
                                            i-th input image.  Specify `targets` during training
                                            only.
            num_classes (int):          The number of classes.

        Returns:
            gt_classes (torch.Tensor):          An integer tensor of shape (N, R) storing
                                                    ground-truth labels for each anchor.
                                                    R is the total number of anchors, i.e. the
                                                    sum of Hi x Wi x A for all levels.
                                                    Anchors with an IoU with some target higher
                                                    than the foreground threshold are assigned
                                                    their corresponding label in the [0, K-1]
                                                    range.
                                                    Anchors whose IoU are below the background
                                                    threshold are assigned the label "K".
                                                    Anchors whose IoU are between the foreground
                                                    and background thresholds are assigned a
                                                    label "-1", i.e. ignore.
            gt_anchors_deltas (torch.Tensor):   Shape (N, R, 4). The last dimension represents
                                                    ground-truth box2box transform targets
                                                    (dx, dy, dw, dh) that map each anchor to its
                                                    matched ground-truth box.
                                                    The values in the tensor are meaningful only
                                                    when the corresponding anchor is labeled as
                                                    foreground.
        """
        output_gt_classes = []
        output_gt_anchors_deltas = []
        anchors = Boxes.cat(anchors)  # Rx4
        # print("############")
        # if num_classes == 1:
            # print("This is a panel")
        # else:
            # print("This is a label")

        # print("anchors:", anchors.tensor.shape)
        # print("instances:", gt_instances)


        for image_gt_instances in gt_instances:
            match_quality_matrix = pairwise_iou(image_gt_instances.gt_boxes, anchors)
            gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)

            # print("GT CLASSES:", gt_classes_per_img)
            # print("GT BOXES:", gt_boxes_per_img)

            # print("gt_classes_per_img", gt_classes_per_img.shape)
            # print("gt_boxes_per_img", gt_boxes_per_img.tensor.shape)

            # print("match_quality_matrix", match_quality_matrix.shape)
            # print("gt_matched_idxs", gt_matched_idxs.shape)

            has_gt = len(image_gt_instances) > 0
            if has_gt:
                # ground truth box regression
                matched_gt_boxes = image_gt_instances.gt_boxes[gt_matched_idxs]
                gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(
                    src_boxes=anchors.tensor,
                    target_boxes=matched_gt_boxes.tensor)
                gt_classes_i = image_gt_instances.gt_classes[gt_matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_classes_i[anchor_labels == 0] = num_classes
                # Anchors with label -1 are ignored.
                gt_classes_i[anchor_labels == -1] = -1
                # print("gt_classes_i", gt_classes_i.shape)
                # print("gt_boxes_i", matched_gt_boxes.tensor.shape)

            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs) + num_classes
                gt_anchors_reg_deltas_i = torch.zeros_like(anchors.tensor)

            output_gt_classes.append(gt_classes_i)
            output_gt_anchors_deltas.append(gt_anchors_reg_deltas_i)

        return torch.stack(output_gt_classes), torch.stack(output_gt_anchors_deltas)


    def inference(self,
                  panel_box_cls: List[torch.Tensor],
                  panel_box_delta: List[torch.Tensor],
                  panel_anchors: List[Boxes],
                  label_box_cls: List[torch.Tensor],
                  label_box_delta: List[torch.Tensor],
                  label_anchors: List[Boxes],
                  image_sizes: List[torch.Size]) -> List[Tuple[Instances, Instances]]:
        """
        Perform inference using the raw batched outputs from the heads.

        Args:
            panel_box_cls (List[torch.Tensor]):     Output class predictions from the panel head.
            panel_box_delta (List[torch.Tensor]):   Output box delta predictions from the panel
                                                        head.
            panel_anchors (List[Boxes]):            A list of #feature level Boxes.
                                                        The Boxes contain anchors of this image on
                                                        the specific feature level.
            label_box_cls (List[torch.Tensor]):     Output class predictions from the label head.
            label_box_delta (List[torch.Tensor]):   Output box delta predictions from the label
                                                        head.
            label_anchors (List[Boxes]):            A list of #feature level Boxes.
                                                        The Boxes contain anchors of this image on
                                                        the specific feature level.
            image_sizes (List[torch.Size]):         The input image sizes.

        Returns:
            results (List[Tuple[Instances, Instances]]):    A list of #images elements.
        """
        results = []

        panel_box_cls = [permute_to_N_HWA_K(x, K=1) for x in panel_box_cls]
        panel_box_delta = [permute_to_N_HWA_K(x, K=4) for x in panel_box_delta]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)
        label_box_cls = [permute_to_N_HWA_K(x, K=self.num_label_classes) for x in label_box_cls]
        label_box_delta = [permute_to_N_HWA_K(x, K=4) for x in label_box_delta]

        for img_idx, image_size in enumerate(image_sizes):
            # Panels
            panel_box_cls_per_image = [box_cls_per_level[img_idx]
                                       for box_cls_per_level in panel_box_cls]
            panel_box_reg_per_image = [box_reg_per_level[img_idx]
                                       for box_reg_per_level in panel_box_delta]

            panel_results = self._inference_single_image(box_cls=panel_box_cls_per_image,
                                                         box_delta=panel_box_reg_per_image,
                                                         anchors=panel_anchors,
                                                         image_size=tuple(image_size),
                                                         num_classes=1)

            # Labels
            label_box_cls_per_image = [box_cls_per_level[img_idx]
                                       for box_cls_per_level in label_box_cls]
            label_box_reg_per_image = [box_reg_per_level[img_idx]
                                       for box_reg_per_level in label_box_delta]

            label_results = self._inference_single_image(box_cls=label_box_cls_per_image,
                                                         box_delta=label_box_reg_per_image,
                                                         anchors=label_anchors,
                                                         image_size=tuple(image_size),
                                                         num_classes=self.num_label_classes)

            results.append((panel_results, label_results))

        return results


    def _filter_detections(self,
                           box_cls: List[torch.Tensor],
                           box_delta: List[torch.Tensor],
                           anchors: List[Boxes],
                           num_classes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Filters raw detections to discard unlikely/poor ones.

        Args:
            box_cls (List[torch.Tensor]):   List of #feature levels. Each entry contains tensor of
                                                size (H x W x A, K).
            box_delta (List[torch.Tensor]): Same shape as 'box_cls' except that K becomes 4.
            anchors (List[Boxes]):          List of #feature levels. Each entry contains a Boxes
                                                object, which contains all the anchors for that
                                                image in that feature level.
            num_classes (int):              The number of classes.

        Returns:
            boxes (torch.Tensor): TODO
            scores (torch.Tensor):
            classes (torch.Tensor):
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // num_classes
            classes_idxs = topk_idxs % num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(deltas=box_reg_i,
                                                                  boxes=anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]

        keep = batched_nms(boxes=boxes_all,
                           scores=scores_all,
                           idxs=class_idxs_all,
                           iou_threshold=self.nms_threshold)

        # Second nms to avoid labels from different classes to overlap
        keep = batched_nms(boxes=boxes_all,
                           scores=scores_all,
                           idxs=torch.ones(size=class_idxs_all.shape),
                           iou_threshold=self.nms_threshold)

        keep = keep[: self.max_detections_per_image]

        return boxes_all[keep], scores_all[keep], class_idxs_all[keep]


    def _inference_single_image(self,
                                box_cls: List[torch.Tensor],
                                box_delta: List[torch.Tensor],
                                anchors: List[Boxes],
                                image_size: Tuple[int, int],
                                num_classes: int
                                ) -> Instances:
        """
        Single-image inference.
        Return bounding-box detection results by thresholding on scores and applying non-maximum
        suppression (NMS).

        Args:
            box_cls (List[torch.Tensor]):   List of #feature levels. Each entry contains tensor of
                                                size (H x W x A, K).
            box_delta (List[torch.Tensor]): Same shape as 'box_cls' except that K becomes 4.
            anchors (List[Boxes]):          List of #feature levels. Each entry contains a Boxes
                                                object, which contains all the anchors for that
                                                image in that feature level.
            image_size (Tuple[int, int]):   A tuple of the image height and width (H, W).
            num_classes (int):              The number of classes.

        Returns:
            results (Instances):    Inference results in an Instances object.
        """

        results = Instances(image_size)
        boxes, scores, class_idxs = self._filter_detections(box_cls=box_cls,
                                                            box_delta=box_delta,
                                                            anchors=anchors,
                                                            num_classes=num_classes)

        results.pred_boxes = Boxes(boxes)
        results.scores = scores
        results.pred_classes = class_idxs

        return results


    def preprocess_image(self, batched_inputs: List[dict]) -> ImageList:
        """
        Normalize, pad and batch the input images.

        Args:
            batched_inputs (List[dict]):    A list, batched outputs of :class:`DatasetMapper`.
                                                Each item in the list contains the inputs for one
                                                image.

        Returns:
            images (ImageList): The transformed images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.panel_fpn.size_divisibility)

        return images


class RetinaNetHead(nn.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.

    Attributes:
        cls_subnet (nn.Module):     Classification subnetwork.
        bbox_subnet (nn.Module):    Regression subnetwork.
        cls_score (nn.Module):      Convolutional layer outputing scores.
        bbox_pred (nn.Module):      Convolutional layer outputing bbox positions.
    """

    def __init__(self,
                 cfg: CfgNode,
                 input_shape: List[ShapeSpec],
                 num_classes: int,
                 num_anchors: int):
        super().__init__()

        in_channels = input_shape[0].channels
        num_convs = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB

        assert len(set(num_anchors)) == 1, "Using different number of anchors between levels is"\
                                           " not currently supported!"
        num_anchors = num_anchors[0]

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(in_channels,
                                   num_anchors
                                   * num_classes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

        self.bbox_pred = nn.Conv2d(in_channels,
                                   num_anchors * 4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)


    def forward(self,
                features: List[torch.Tensor]
                ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            features (list[Tensor]):    FPN feature map tensors in high to low resolution.
                                            Each tensor in the list correspond to different
                                            feature levels.

        Returns:
            logits (list[torch.Tensor]):    #lvl tensors, each has shape (N, AxK, Hi, Wi).
                                                The tensor predicts the classification probability
                                                at each spatial position for each of the A anchors
                                                and K object classes.
            bbox_reg (list[torch.Tensor]):  #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                                                The tensor predicts 4-vector (dx,dy,dw,dh) box
                                                regression values for every anchor. These values
                                                are the relative offset between the anchor and the
                                                ground truth box.
        """
        logits = []
        bbox_reg = []

        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))

        return logits, bbox_reg
