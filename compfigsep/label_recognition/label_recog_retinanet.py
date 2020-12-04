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


##########################################################
Custom variation of RetinaNet to handle label recognition.

[WiP]
"""

import math
from typing import List, Dict, Tuple, Any
import torch
from torch import nn, Tensor, LongTensor
from torch.nn import functional as F
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss

from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.config import CfgNode

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone.resnet import build_resnet_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.postprocessing import detector_postprocess

from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K

from detectron2.modeling.backbone.fpn import FPN


__all__ = ["LabelRecogRetinaNet"]



def build_fpn_backbone(cfg: CfgNode,
                       input_shape: ShapeSpec) -> nn.Module:
    """
    Build the Resnet 50 backbone and the FPN:
    Label FPN:
      * Input:  C2, C3, C4
      * Output: P2, P3, P4

    Args:
        cfg (CfgNode):              A detectron2 CfgNode
        input_shape (ShapeSpec):    The input shape of the backbone.

    Returns:
        backbone (nn.Module):   Backbone module, must be a subclass of :class:`Backbone`.
    """
    # Build the feature extractor (Resnet 50)
    bottom_up: nn.Module = build_resnet_backbone(cfg, input_shape)

    # Label FPN
    label_in_features: List[str] = cfg.MODEL.FPN.IN_FEATURES
    label_out_channels: List[str] = cfg.MODEL.FPN.OUT_CHANNELS
    label_fpn: nn.Module = FPN(bottom_up=bottom_up,
                               in_features=label_in_features,
                               out_channels=label_out_channels,
                               norm=cfg.MODEL.FPN.NORM,
                               top_block=None,
                               fuse_type=cfg.MODEL.FPN.FUSE_TYPE)

    return label_fpn


class LabelRecogRetinaNet(nn.Module):
    """
    Implement RetinaNet (https://arxiv.org/abs/1708.02002).
    """

    def __init__(self, cfg) -> None:
        super().__init__()

        self.num_classes: int = cfg.MODEL.RETINANET.NUM_CLASSES
        self.in_features: List[str] = cfg.MODEL.RETINANET.IN_FEATURES
        # Loss parameters:
        self.focal_loss_alpha: float = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma: float = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta: float = cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA
        # Inference parameters:
        self.score_threshold: float = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
        self.topk_candidates: int = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
        self.nms_threshold: float = cfg.MODEL.RETINANET.NMS_THRESH_TEST
        self.max_detections_per_image: int = cfg.TEST.DETECTIONS_PER_IMAGE
        # Vis parameters
        self.vis_period: int = cfg.VIS_PERIOD
        self.input_format: str = cfg.INPUT.FORMAT

        self.fpn: FPN = build_fpn_backbone(
            cfg,
            input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        backbone_fpn_output_shape: Dict[str, ShapeSpec] = self.fpn.output_shape()

        feature_shapes: List[ShapeSpec] = [backbone_fpn_output_shape[f]
                                           for f in self.in_features]
        self.head: RetinaNetHead = RetinaNetHead(cfg, feature_shapes)

        self.anchor_generator: nn.Module = build_anchor_generator(cfg, feature_shapes)

        # Matching and loss
        self.box2box_transform: Box2BoxTransform = Box2BoxTransform(
            weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)

        self.anchor_matcher: Matcher = Matcher(
            thresholds=cfg.MODEL.RETINANET.IOU_THRESHOLDS,
            labels=cfg.MODEL.RETINANET.IOU_LABELS,
            allow_low_quality_matches=True)

        self.register_buffer("pixel_mean",
                             torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std",
                             torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

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
        Returns:
        TODO
        """
        return self.pixel_mean.device


    def forward(self, batched_inputs: List[dict]) -> Union[Dict[str, Any],
                                                           List[Dict[str, Instances]]]:
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            dict[str: Tensor]:  Mapping from a named loss to a tensor storing the loss.
                                    Used during training only.
        """
        images: ImageList = self.preprocess_image(batched_inputs)


        features_dict: Dict[str, ShapeSpec] = self.fpn(images.tensor)
        features: List[ShapeSpec] = [features_dict[f] for f in self.in_features]
        pred_logits, pred_anchor_deltas = self.head(features)
        anchors: List[Boxes] = self.anchor_generator(features)

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, K=self.num_classes)
                       for x in pred_logits]

        pred_anchor_deltas = [permute_to_N_HWA_K(x, K=4)
                              for x in pred_anchor_deltas]

        if self.training:
            gt_instances: Instances = [x['instances'].to(self.device) for x in batched_inputs]
            gt_classes, gt_boxes = self.get_ground_truth(anchors=anchors,
                                                         gt_instances=gt_instances)
            losses = self.losses(anchors=anchors,
                                 pred_logits=pred_logits,
                                 gt_classes=gt_classes,
                                 pred_anchor_deltas=pred_anchor_deltas,
                                 gt_boxes=gt_boxes)

            return losses

        # Otherwise, do inference.
        results = self.inference(anchors=anchors,
                                 pred_logits=pred_logits,
                                 pred_anchor_deltas=pred_anchor_deltas,
                                 image_sizes=images.image_sizes)

        processed_results: List[Dict[str, Any]] = []
        for results_per_image, input_per_image, image_size in zip(results,
                                                                  batched_inputs,
                                                                  images.image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({"instances": detector_postprocess(results_per_image,
                                                                        height,
                                                                        width)})
        return processed_results


    def losses(self,
               anchors: List[Boxes],
               pred_logits: List[Tensor],
               gt_classes: List[Tensor],
               pred_anchor_deltas: List[Tensor],
               gt_boxes: List[Tensor]) -> Dict[str, float]:
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`RetinaNetHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
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
        pos_mask: Tensor = (gt_classes_tensor >= 0) & (gt_classes_tensor != self.num_classes)
        num_pos_anchors: int = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors",
                                       num_pos_anchors / num_images)
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer\
                                + (1 - self.loss_normalizer_momentum) * max(num_pos_anchors, 1)

        # classification and regression loss
        # no loss for the last (background) class --> [:, :-1]
        gt_classes_target: LongTensor = F.one_hot(gt_classes_tensor[valid_mask],
                                                  num_classes=self.num_classes + 1)[:, :-1]

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(inputs=cat(pred_logits, dim=1)[valid_mask],
                                          targets=gt_classes_target.to(pred_logits[0].dtype),
                                          alpha=self.focal_loss_alpha,
                                          gamma=self.focal_loss_gamma,
                                          reduction="sum") / self.loss_normalizer

        # regression loss
        loss_box_reg = smooth_l1_loss(input=cat(pred_anchor_deltas, dim=1)[pos_mask],
                                      target=gt_anchor_deltas_tensor[pos_mask],
                                      beta=self.smooth_l1_loss_beta,
                                      reduction="sum") / self.loss_normalizer

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}


    @torch.no_grad()
    def get_ground_truth(self,
                         anchors: List[Boxes],
                         gt_instances: List[Instances]) -> Tuple[List[Tensor],
                                                                 List[Tensor]]:
        """
        Args:
            anchors (List[Boxes]):      A list of #feature level Boxes.
                                            The Boxes contains anchors of this image on the
                                            specific feature level.
            targets (list[Instances]):  A list of N `Instances`s.
                                            The i-th `Instances` contains the ground-truth
                                            per-instance annotations for the i-th input image.
                                            Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):        An integer tensor of shape (N, R) storing ground-truth
                                            labels for each anchor.
                                            R is the total number of anchors, i.e. the sum of
                                            Hi x Wi x A for all levels.
                                            Anchors with an IoU with some target higher than the
                                            foreground threshold are assigned their corresponding
                                            label in the [0, K-1] range. Anchors whose IoU are
                                            below the background threshold are assigned the label
                                            "K".
                                            Anchors whose IoU are between the foreground and
                                            background thresholds are assigned a label "-1", i.e.
                                            ignore.
            gt_anchors_deltas (Tensor): Shape (N, R, 4).
                                            The last dimension represents ground-truth box2box
                                            transform targets (dx, dy, dw, dh) that map each
                                            anchor to its matched ground-truth box.
                                            The values in the tensor are meaningful only when the
                                            corresponding anchor is labeled as foreground.
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
                gt_classes_i[anchor_classes == 0] = self.num_classes
                # Anchors with class -1 are ignored.
                gt_classes_i[anchor_classes == -1] = -1
                # TODO remove
                # print("gt_labels_i", gt_classes_i.shape)
                # print("gt_boxes_i", matched_gt_boxes.tensor.shape)

            else:
                matched_gt_boxes_i = torch.zeros_like(anchors_boxes.tensor)
                gt_classes_i = torch.zeros_like(matched_idxs) + self.num_classes

            gt_classes.append(gt_classes_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_classes, matched_gt_boxes


    def inference(self,
                  anchors: List[Boxes],
                  pred_logits: List[Tensor],
                  pred_anchor_deltas: List[Tensor],
                  image_sizes: List[torch.Size]) -> List[Instances]:
        """
        Args:
            box_cls, box_delta:             Same as the output of :meth:`RetinaNetHead.forward`
            anchors (list[Boxes]):          A list of #feature level Boxes.
                                                The Boxes contain anchors of this image on the
                                                specific feature level.
            image_sizes (List[torch.Size]): The input image sizes

        Returns:
            results (List[Instances]):  A list of #images elements.
        """
        results: List[Instances] = []

        for img_idx, image_size in enumerate(image_sizes):
            pred_logits_per_image: List[Tensor] = [x[img_idx] for x in pred_logits]
            deltas_per_image: List[Tensor] = [x[img_idx] for x in pred_anchor_deltas]
            results_per_image = self._inference_single_image(anchors=anchors,
                                                             box_cls=pred_logits_per_image,
                                                             box_delta=deltas_per_image,
                                                             image_size=tuple(image_size))
            results.append(results_per_image)

        return results


    def _inference_single_image(self,
                                anchors: List[Boxes],
                                box_cls: List[Tensor],
                                box_delta: List[Tensor],
                                image_size: Tuple[int, int]) -> Instances:
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Args:
            box_cls (list[Tensor]):     List of #feature levels. Each entry contains tensor of
                                            size (H x W x A, K).
            box_delta (list[Tensor]):   Same shape as 'box_cls' except that K becomes 4.
            anchors (list[Boxes]):      List of #feature levels. Each entry contains a Boxes
                                            object, which contains all the anchors for that image
                                            in that feature level.
            image_size (tuple(H, W)):   A tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all: List[Tensor] = []
        scores_all: List[Tensor] = []
        class_idxs_all: List[Tensor] = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k top scoring indices only.
            num_topk: int = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i,
                                                                  anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [cat(x)
                                                 for x in [boxes_all,
                                                           scores_all,
                                                           class_idxs_all]]

        keep = batched_nms(boxes=boxes_all,
                           scores=scores_all,
                           idxs=class_idxs_all,
                           iou_threshold=self.nms_threshold)

        # Second nms to avoid labels from different classes to overlap
        keep = batched_nms(boxes=boxes_all,
                           scores=scores_all,
                           idxs=torch.ones(size=class_idxs_all.shape),
                           iou_threshold=self.nms_threshold)

        keep = keep[:self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result


    def preprocess_image(self,
                         batched_inputs: List[Dict[str, Any]]) -> ImageList:
        """
        Normalize, pad and batch the input images.

        Args:
            batched_inputs (List[Dict[str ,Any]]):  List of batched inputs.

        Returns:
            images (ImageList): An ImageList structure containing the preprocessed image data.
        """
        images: List[Tensor] = [x["image"].to(self.device)
                                for x in batched_inputs]

        images = [(x - self.pixel_mean) / self.pixel_std
                  for x in images]

        images: ImageList = ImageList.from_tensors(images, self.fpn.size_divisibility)

        return images


# TODO import from detectron2 directly as we should not need to change the head.
class RetinaNetHead(nn.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self,
                 cfg: CfgNode,
                 input_shape: List[ShapeSpec]) -> None:
        super().__init__()
        in_channels = input_shape[0].channels
        num_classes: int = cfg.MODEL.RETINANET.NUM_CLASSES
        num_convs: int = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob: float = cfg.MODEL.RETINANET.PRIOR_PROB
        num_anchors: List[int] = build_anchor_generator(cfg,
                                                        input_shape).num_cell_anchors

        assert len(set(num_anchors)) == 1,\
            "Using different number of anchors between levels is not currently supported!"

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


    def forward(self, features: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits: List[Tensor] = []
        bbox_reg: List[Tensor] = []

        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg
