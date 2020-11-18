"""
#############################
#        CompFigSep         #
# Compound Figure Separator #
#############################

GitHub:         https://github.com/GaetanLepage/compound-figure-separator

Author:         Gaétan Lepage
Email:          gaetan.lepage@grenoble-inp.fr
Date:           Spring 2020

Master's project @HES-SO (Sierre, SW)

Supervisors:    Henning Müller (henning.mueller@hevs.ch)
                Manfredo Atzori (manfredo.atzori@hevs.ch)

Collaborators:  Niccolò Marini (niccolo.marini@hevs.ch)
                Stefano Marchesin (stefano.marchesin@unipd.it)


##################################################################
Custom model derived from RetinaNet to achieve panel segmentation.
"""

import math
from typing import cast, List, Tuple, Dict, Any, Union
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss # type: ignore
from torch import nn, Tensor, LongTensor
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, batched_nms, cat # type: ignore
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou # type: ignore
from detectron2.utils.events import get_event_storage # type: ignore
from detectron2.config import CfgNode # type: ignore

from detectron2.modeling.anchor_generator import DefaultAnchorGenerator # type: ignore
from detectron2.modeling.backbone.resnet import build_resnet_backbone # type: ignore
from detectron2.modeling.box_regression import Box2BoxTransform # type: ignore
from detectron2.modeling.matcher import Matcher # type: ignore

from detectron2.modeling.backbone.fpn import FPN, LastLevelP6P7 # type: ignore
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K # type: ignore

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
        cfg (CfgNode):              A detectron2 CfgNode.
        input_shape (ShapeSpec):    The input shape of the backbone.

    Returns:
        backbone (Backbone):    Backbone module, must be a subclass of :class:`Backbone`.
    """
    # Build the feature extractor (Resnet 50)
    bottom_up: nn.Module = build_resnet_backbone(cfg, input_shape)

    # Panel FPN
    panel_in_features: List[str] = cfg.MODEL.PANEL_FPN.IN_FEATURES
    panel_out_channels: List[str] = cfg.MODEL.PANEL_FPN.OUT_CHANNELS
    in_channels_p6p7: int = bottom_up.output_shape()['res5'].channels
    panel_fpn: nn.Module = FPN(bottom_up=bottom_up,
                               in_features=panel_in_features,
                               out_channels=panel_out_channels,
                               norm=cfg.MODEL.FPN.NORM,
                               top_block=LastLevelP6P7(in_channels_p6p7,
                                                       panel_out_channels),
                               fuse_type=cfg.MODEL.FPN.FUSE_TYPE)


    # Label FPN
    label_in_features: List[str] = cfg.MODEL.LABEL_FPN.IN_FEATURES
    label_out_channels: List[str] = cfg.MODEL.LABEL_FPN.OUT_CHANNELS
    label_fpn: nn.Module = FPN(bottom_up=bottom_up,
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
        anchor_matcher (Matcher):               Match predicted element to a ground-truth element.
        loss_normalizer (float):                Quantity to divide the loss by.
        loss_normalizer_momentum (float):       The loss normalizer momentum.
    """

    def __init__(self, cfg: CfgNode):
        super().__init__()

        self.num_label_classes: int = cfg.MODEL.RETINANET.NUM_LABEL_CLASSES
        self.panel_in_features: List[str] = cfg.MODEL.RETINANET.PANEL_IN_FEATURES
        self.label_in_features: List[str] = cfg.MODEL.RETINANET.LABEL_IN_FEATURES
        # Loss parameters:
        self.focal_loss_alpha: float = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma: float = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta: float = cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA
        # Inference parameters:
        self.score_threshold: float = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
        self.topk_candidates: int = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
        self.nms_threshold: float = cfg.MODEL.RETINANET.NMS_THRESH_TEST
        self.max_detections_per_image: int = cfg.TEST.DETECTIONS_PER_IMAGE

        self.panel_fpn, self.label_fpn = build_fpn_backbones(
            cfg,
            input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        # Panel
        panel_backbone_fpn_output_shape: Dict[str, ShapeSpec] = self.panel_fpn.output_shape()
        panel_feature_shapes: List[ShapeSpec] = [panel_backbone_fpn_output_shape[f]
                                                 for f in self.panel_in_features]

        self.panel_anchor_generator: DefaultAnchorGenerator = DefaultAnchorGenerator(
            sizes=cfg.MODEL.PANEL_ANCHOR_GENERATOR.SIZES,
            aspect_ratios=cfg.MODEL.PANEL_ANCHOR_GENERATOR.ASPECT_RATIOS,
            strides=[x.stride for x in panel_feature_shapes],
            offset=cfg.MODEL.ANCHOR_GENERATOR.OFFSET)

        self.panel_head: RetinaNetHead = RetinaNetHead(
            cfg,
            input_shape=panel_feature_shapes,
            num_classes=1,
            num_anchors=self.panel_anchor_generator.num_cell_anchors)


        # Label
        label_backbone_fpn_output_shape: Dict[str, ShapeSpec] = self.label_fpn.output_shape()
        label_feature_shapes: List[ShapeSpec] = [label_backbone_fpn_output_shape[f]
                                                 for f in self.label_in_features]

        self.label_anchor_generator: DefaultAnchorGenerator = DefaultAnchorGenerator(
            sizes=cfg.MODEL.LABEL_ANCHOR_GENERATOR.SIZES,
            aspect_ratios=cfg.MODEL.LABEL_ANCHOR_GENERATOR.ASPECT_RATIOS,
            strides=[x.stride for x in label_feature_shapes],
            offset=cfg.MODEL.ANCHOR_GENERATOR.OFFSET)

        self.label_head: RetinaNetHead = RetinaNetHead(
            cfg,
            input_shape=label_feature_shapes,
            num_classes=cfg.MODEL.RETINANET.NUM_LABEL_CLASSES,
            num_anchors=self.label_anchor_generator.num_cell_anchors)

        # Matching and loss
        # TODO maybe have to duplicate those as well
        self.box2box_transform: Box2BoxTransform = Box2BoxTransform(
            weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher: Matcher = Matcher(cfg.MODEL.RETINANET.IOU_THRESHOLDS,
                                               cfg.MODEL.RETINANET.IOU_LABELS,
                                               allow_low_quality_matches=True)

        self.register_buffer("pixel_mean",
                             Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std",
                             Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        # In Detectron1, loss is normalized by number of foreground samples in the batch.
        # When batch size is 1 per GPU, #foreground has a large variance and
        # using it lead to lower performance. Here we maintain an EMA of #foreground to
        # stabilize the normalizer.

        # Initialize with any reasonable #fg that's not too small
        self.loss_normalizer: float = 100
        self.loss_normalizer_momentum: float = 0.9


    @property
    def device(self) -> torch.device:
        """
        Identify the device the model is hosted by.

        Returns:
            device (torch.device):  The model device.
        """
        return self.pixel_mean.device


    def forward(self, batched_inputs: List[dict]) -> Union[Dict[str, Any],
                                                           List[Dict[str, Instances]]]:
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
        images: ImageList = self.preprocess_image(batched_inputs)

        # detected panels
        panel_features_dict: Dict[str, ShapeSpec] = self.panel_fpn(images.tensor)
        panel_features: List[ShapeSpec] = [panel_features_dict[f] for f in self.panel_in_features]
        panel_anchors: List[Boxes] = self.panel_anchor_generator(panel_features)
        panel_pred_logits, panel_pred_anchor_deltas = self.panel_head(panel_features)
        # Transpose the Hi*Wi*A dimension to the middle:
        panel_pred_logits = [permute_to_N_HWA_K(x, K=1)
                             for x in panel_pred_logits]

        panel_pred_anchor_deltas = [permute_to_N_HWA_K(x, K=4)
                                    for x in panel_pred_anchor_deltas]

        # detected labels
        label_features_dict: Dict[str, ShapeSpec] = self.label_fpn(images.tensor)
        label_features: List[ShapeSpec] = [label_features_dict[f]
                                           for f in self.label_in_features]
        label_anchors: List[Boxes] = self.label_anchor_generator(label_features)
        label_pred_logits, label_pred_anchor_deltas = self.label_head(label_features)
        # Transpose the Hi*Wi*A dimension to the middle:
        label_pred_logits = [permute_to_N_HWA_K(x, K=self.num_label_classes)
                             for x in label_pred_logits]

        label_pred_anchor_deltas = [permute_to_N_HWA_K(x, K=4)
                                    for x in label_pred_anchor_deltas]

        # Training
        if self.training:

            # Panels
            panel_gt_instances: Instances = [x['panel_instances'].to(self.device)
                                             for x in batched_inputs]

            panel_gt_classes, panel_gt_boxes = self.get_ground_truth(
                anchors=panel_anchors,
                gt_instances=panel_gt_instances,
                num_classes=1)

            panel_loss_cls, panel_loss_box_reg = self._compute_single_head_losses(
                anchors=panel_anchors,
                pred_logits=panel_pred_logits,
                gt_classes=panel_gt_classes,
                pred_anchor_deltas=panel_pred_anchor_deltas,
                gt_boxes=panel_gt_boxes,
                num_classes=1)


            loss_dict: Dict[str, float] = {
                'panel_loss_cls': panel_loss_cls,
                'panel_loss_box_reg': panel_loss_box_reg
            }

            # Labels
            label_gt_instances: Instances = [x['label_instances'].to(self.device)
                                             for x in batched_inputs]

            label_gt_classes, label_gt_boxes = self.get_ground_truth(
                anchors=label_anchors,
                gt_instances=label_gt_instances,
                num_classes=self.num_label_classes)

            label_loss_cls, label_loss_box_reg = self._compute_single_head_losses(
                anchors=label_anchors,
                pred_logits=label_pred_logits,
                gt_classes=label_gt_classes,
                pred_anchor_deltas=label_pred_anchor_deltas,
                gt_boxes=label_gt_boxes,
                num_classes=self.num_label_classes)

            loss_dict['label_loss_cls'] = label_loss_cls
            loss_dict['label_loss_box_reg'] = label_loss_box_reg

            return loss_dict

        # Otherwise, do inference.
        batched_inference_results = self.inference(
            panel_anchors=panel_anchors,
            panel_pred_logits=panel_pred_logits,
            panel_pred_anchor_deltas=panel_pred_anchor_deltas,
            label_anchors=label_anchors,
            label_pred_logits=label_pred_logits,
            label_pred_anchor_deltas=label_pred_anchor_deltas,
            image_sizes=images.image_sizes)

        processed_results: List[Dict[str, Instances]] = []

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
            label_results = Instances((height,
                                       width),
                                      **label_results.get_fields())

            # Clip and scale boxes
            label_output_boxes = label_results.pred_boxes
            label_output_boxes.scale(scale_x, scale_y)
            label_output_boxes.clip(label_results.image_size)

            processed_results.append({"panels": panel_results, "labels": label_results})

        return processed_results


    def _compute_single_head_losses(self,
                                    anchors: List[Boxes],
                                    pred_logits: List[Tensor],
                                    gt_classes: List[Tensor],
                                    pred_anchor_deltas: List[Tensor],
                                    gt_boxes: List[Tensor],
                                    num_classes: int) -> Tuple[float, float]:
        """
        Compute the loss dict for a single RetinaNet branch (one classification head and one
        regression head).

        N is the number of images in the batch.
        R is the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)

        Args:
            anchors (List[Boxes]):              A list of #features level Boxes.
            pred_logits (List[Tensor]):         Predicted classes for the batch images.
                                                    shape=(N, Hi * Wi * Ai, num_classes)
            gt_classes (List[Tensor]):          Ground truth classes. List of N tensors.
            pred_anchor_deltas (List[Tensor]):  Predicted box deltas.
                                                    shape=(N, Hi * Wi * Ai, 4)
            gt_boxes (List[Tensor]):            Ground truth box deltas. List of N tensors of
                                                    shape=(R, 4).
            num_classes (int):                  The number of classes.

        Returns:
            loss_cls, loss_box_reg (Tuple[float, float]):   The value of the loss for each head
                                                                (classification and regression).
        """
        num_images: int = len(gt_classes)

        # shape(gt_classes) = (N, R)
        gt_classes_tensor: Tensor = torch.stack(gt_classes)

        # shape(anchors) = (R, 4)
        anchors_tensor: Tensor = type(anchors[0]).cat(anchors).tensor
        gt_anchor_deltas: List[Tensor] = [self.box2box_transform.get_deltas(anchors_tensor, k)
                                          for k in gt_boxes]
        # shape(gt_anchor_deltas) = (N, R, 4)
        gt_anchor_deltas_tensor: Tensor = torch.stack(gt_anchor_deltas)

        valid_mask: Tensor = gt_classes_tensor >= 0
        pos_mask: Tensor = (gt_classes_tensor >= 0) & (gt_classes_tensor != num_classes)
        num_pos_anchors: int = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors",
                                       num_pos_anchors / num_images)
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer\
                                + (1 - self.loss_normalizer_momentum) * max(num_pos_anchors, 1)

        # classification and regression loss
        # no loss for the last (background) class --> [:, :-1]
        gt_labels_target: LongTensor = F.one_hot(gt_classes_tensor[valid_mask],
                                                 num_classes=num_classes + 1)[:, :-1]

        # logits loss
        loss_cls: float = sigmoid_focal_loss_jit(
            inputs=cat(pred_logits, dim=1)[valid_mask],
            targets=gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum") / self.loss_normalizer

        # regression loss
        loss_box_reg: float = smooth_l1_loss(
            input=cat(pred_anchor_deltas, dim=1)[pos_mask],
            target=gt_anchor_deltas_tensor[pos_mask],
            beta=self.smooth_l1_loss_beta,
            reduction="sum") / self.loss_normalizer

        return loss_cls, loss_box_reg


    @torch.no_grad()
    def get_ground_truth(self,
                         anchors: List[Boxes],
                         gt_instances: List[Instances],
                         num_classes: int) -> Tuple[List[Tensor],
                                                    List[Tensor]]:
        """
        Extract the ground truth classes and boxes from a list of Instances objects.

        Args:
            anchors (List[Boxes]):          A list of #feature level Boxes. The Boxes contains
                                                anchors of this image on the specific feature
                                                level.
            gt_instances (List[Instances]): A list of N `Instances`s. The i-th `Instances`
                                                contains the ground-truth per-instance annotations
                                                for the i-th input image.
            num_classes (int):              The number of classes.

        Returns:
            gt_classes (List[Tensor]):          List of #img tensors. i-th element is a vector of
                                                    classes whose length is the total number of
                                                    anchors across all feature maps
                                                    (sum(Hi * Wi * A)).
                                                    Label values are in {-1, 0, ..., K}, with -1
                                                    means ignore, and K means background.
            matched_gt_boxes (List[Tensor]):    i-th element is a Rx4 tensor, where R is the total
                                                    number of anchors across feature maps.
                                                    The values are the matched gt boxes for each
                                                    anchor.
                                                    Values are undefined for those anchors not
                                                    labeled as foreground.
        """
        anchors_boxes: Boxes = Boxes.cat(anchors)

        gt_classes: List[Tensor] = []
        matched_gt_boxes: List[Tensor] = []

        for gt_instance in gt_instances:
            match_quality_matrix: Tensor = pairwise_iou(gt_instance.gt_boxes,
                                                        anchors_boxes)
            matched_idxs, anchor_classes = self.anchor_matcher(match_quality_matrix)
            del match_quality_matrix

            if len(gt_instance) > 0:
                matched_gt_boxes_i: Tensor = gt_instance.gt_boxes.tensor[matched_idxs]

                gt_classes_i: Tensor = gt_instance.gt_classes[matched_idxs]

                # Anchors with class 0 are treated as background.
                gt_classes_i[anchor_classes == 0] = num_classes
                # Anchors with class -1 are ignored.
                gt_classes_i[anchor_classes == -1] = -1

            else:
                matched_gt_boxes_i = torch.zeros_like(anchors_boxes.tensor)
                gt_classes_i = torch.zeros_like(matched_idxs) + num_classes

            gt_classes.append(gt_classes_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_classes, matched_gt_boxes


    def inference(self,
                  panel_anchors: List[Boxes],
                  panel_pred_logits: List[Tensor],
                  panel_pred_anchor_deltas: List[Tensor],
                  label_anchors: List[Boxes],
                  label_pred_logits: List[Tensor],
                  label_pred_anchor_deltas: List[Tensor],
                  image_sizes: List[torch.Size]) -> List[Tuple[Instances, Instances]]:
        """
        Perform inference using the raw batched outputs from the heads.

        Args:
            panel_anchors (List[Boxes]):                A list of #feature level Boxes.
                                                            The Boxes contain anchors of this
                                                            image on the specific feature level.
            panel_pred_logits (List[Tensor]):           Output class predictions from the panel
                                                            head.
                                                            One tensor per level.
                                                            shape=(N, Hi * Wi * Ai, 1)
            panel_pred_anchor_deltas (List[Tensor]):    Output box delta predictions from the
                                                            panel head.
                                                            One tensor per level.
                                                            shape=(N, Hi * Wi * Ai, 4)
            label_anchors (List[Boxes]):                A list of #feature level Boxes.
                                                            The Boxes contain anchors of this
                                                            image on the specific feature level.
            label_pred_logits (List[Tensor]):           Output class predictions from the label
                                                            head.
                                                            One tensor per level.
                                                            shape=(N, Hi * Wi * Ai, num_classes)
            label_pred_anchor_deltas (List[Tensor]):    Output box delta predictions from the
                                                            label head.
                                                            One tensor per level.
                                                            shape=(N, Hi * Wi * Ai, 4)
            image_sizes (List[torch.Size]):             The input image sizes.

        Returns:
            results (List[Tuple[Instances, Instances]]):    A list of #images elements.
        """
        results: List[Tuple[Instances, Instances]] = []

        for img_idx, image_size in enumerate(image_sizes):
            # Panels
            panel_pred_logits_per_image = [x[img_idx]
                                           for x in panel_pred_logits]
            panel_deltas_per_image = [x[img_idx]
                                      for x in panel_pred_anchor_deltas]

            image_size_tuple = cast(Tuple[int, int],
                                    image_size)

            panel_results_per_image = self._inference_single_image(
                anchors=panel_anchors,
                box_cls=panel_pred_logits_per_image,
                box_delta=panel_deltas_per_image,
                image_size=image_size_tuple,
                num_classes=1)

            # Labels
            label_pred_logits_per_image = [x[img_idx]
                                           for x in label_pred_logits]
            label_deltas_per_image = [x[img_idx]
                                      for x in label_pred_anchor_deltas]

            label_results_per_image = self._inference_single_image(
                anchors=label_anchors,
                box_cls=label_pred_logits_per_image,
                box_delta=label_deltas_per_image,
                image_size=image_size_tuple,
                num_classes=self.num_label_classes)

            results.append((panel_results_per_image, label_results_per_image))

        return results


    def _filter_detections(self,
                           box_cls: List[Tensor],
                           box_delta: List[Tensor],
                           anchors: List[Boxes],
                           num_classes: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Filters raw detections to discard unlikely/poor ones.

        Args:
            box_cls (List[Tensor]):   List of #feature levels. Each entry contains tensor of
                                                size (H x W x A, K).
            box_delta (List[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            anchors (List[Boxes]):          List of #feature levels. Each entry contains a Boxes
                                                object, which contains all the anchors for that
                                                image in that feature level.
            num_classes (int):              The number of classes.

        Returns:
            boxes (Tensor):     The filtered bounding boxes.
            scores (Tensor):    The detection scores.
            classes (Tensor):   The detections classes.
        """
        boxes_all: List[Tensor] = []
        scores_all: List[Tensor] = []
        class_idxs_all: List[Tensor] = []

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

        boxes_tensor, scores_tensor, class_idxs_tensor = (cat(x)
                                                          for x in (boxes_all,
                                                                    scores_all,
                                                                    class_idxs_all))

        keep: Tensor = batched_nms(boxes=boxes_tensor,
                                   scores=scores_tensor,
                                   idxs=class_idxs_tensor,
                                   iou_threshold=self.nms_threshold)

        # Second nms to avoid labels from different classes to overlap
        keep = batched_nms(boxes=boxes_tensor,
                           scores=scores_tensor,
                           idxs=torch.ones(size=class_idxs_tensor.shape),
                           iou_threshold=self.nms_threshold)

        keep = keep[: self.max_detections_per_image]

        return boxes_tensor[keep], scores_tensor[keep], class_idxs_tensor[keep]


    def _inference_single_image(self,
                                anchors: List[Boxes],
                                box_cls: List[Tensor],
                                box_delta: List[Tensor],
                                image_size: Tuple[int, int],
                                num_classes: int) -> Instances:
        """
        Single-image inference.
        Return bounding-box detection results by thresholding on scores and applying non-maximum
        suppression (NMS).

        Args:
            anchors (List[Boxes]):          List of #feature levels. Each entry contains a Boxes
                                                object, which contains all the anchors in that
                                                feature level.
            box_cls (List[Tensor]):         List of #feature levels. Each entry contains tensor of
                                                size (H x W x A, K).
            box_delta (List[Tensor]):       Same shape as 'box_cls' except that K becomes 4.
            image_size (Tuple[int, int]):   A tuple of the image height and width (H, W).
            num_classes (int):              The number of classes.

        Returns:
            results (Instances):    Inference results in an Instances object.
        """

        results: Instances = Instances(image_size)
        boxes, scores, class_idxs = self._filter_detections(box_cls=box_cls,
                                                            box_delta=box_delta,
                                                            anchors=anchors,
                                                            num_classes=num_classes)

        results.pred_boxes = Boxes(boxes)
        results.scores = scores
        results.pred_classes = class_idxs

        return results


    def preprocess_image(self,
                         batched_inputs: List[Dict[str,
                                                   Any]]) -> ImageList:
        """
        Normalize, pad and batch the input images.

        Args:
            batched_inputs (List[dict]):    A list, batched outputs of :class:`DatasetMapper`.
                                                Each item in the list contains the inputs for one
                                                image.

        Returns:
            images (ImageList): The transformed images.
        """
        images_list: List[Tensor] = [x['image'].to(self.device)
                                     for x in batched_inputs]
        images_list = [(x - self.pixel_mean) / self.pixel_std
                       for x in images_list]

        images = ImageList.from_tensors(images_list,
                                        self.panel_fpn.size_divisibility)

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
                 num_anchors: List[int]) -> None:

        super().__init__()

        in_channels = input_shape[0].channels
        num_convs: int = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob: float = cfg.MODEL.RETINANET.PRIOR_PROB

        assert len(set(num_anchors)) == 1, "Using different number of anchors between levels is"\
                                           " not currently supported!"

        num_anchors_int: int = num_anchors[0]

        cls_subnet: List[nn.Module] = []
        bbox_subnet: List[nn.Module] = []
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
                                   num_anchors_int
                                   * num_classes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

        self.bbox_pred = nn.Conv2d(in_channels,
                                   num_anchors_int * 4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

        # Initialization
        for modules in [self.cls_subnet,
                        self.bbox_subnet,
                        self.cls_score,
                        self.bbox_pred]:

            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(tensor=layer.weight,
                                          mean=0,
                                          std=0.01)
                    torch.nn.init.constant_(tensor=layer.bias,
                                            val=0)

        # Use prior in model initialization to improve stability
        bias_value: float = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias,
                                bias_value)


    def forward(self,
                features: List[Tensor]) -> Tuple[List[Tensor],
                                                 List[Tensor]]:
        """
        Args:
            features (list[Tensor]):    FPN feature map tensors in high to low resolution.
                                            Each tensor in the list correspond to different
                                            feature levels.

        Returns:
            logits (list[Tensor]):    #lvl tensors, each has shape (N, AxK, Hi, Wi).
                                                The tensor predicts the classification probability
                                                at each spatial position for each of the A anchors
                                                and K object classes.
            bbox_reg (list[Tensor]):  #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                                                The tensor predicts 4-vector (dx,dy,dw,dh) box
                                                regression values for every anchor. These values
                                                are the relative offset between the anchor and the
                                                ground truth box.
        """
        logits: List[Tensor] = []
        bbox_reg: List[Tensor] = []

        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))

        return logits, bbox_reg
