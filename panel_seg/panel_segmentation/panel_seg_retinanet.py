# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import List
import numpy as np
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn

from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone.resnet import build_resnet_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.postprocessing import detector_postprocess

from detectron2.modeling.backbone.fpn import FPN, LastLevelP6P7

__all__ = ["PanelSegRetinaNet"]


# TODO maybe split in multiple files (in a folder model/ for eg)


def build_fpn_backbones(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
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


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)

    Args:
        tensor (torch.Tensor):  An output tensor from a RetinaNetHead.
        K (int):                The number of classes (for classification) or 4 (for regression).
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def permute_all_cls_and_box_to_N_HWA_K_and_concat(box_cls,
                                                  box_delta,
                                                  num_classes: int = 80):
    """
    Rearrange the tensor layout from the network output to per-image predictions.

    Args:
        box_cls, box_delta (list[Tensor]):  #lvl tensors of shape (N, A x K, Hi, Wi)
        num_classes (int):                  The number of classes.

    Returns:
        Tensor of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta


class PanelSegRetinaNet(nn.Module):
    """
    Implement RetinaNet (https://arxiv.org/abs/1708.02002).
    """

    def __init__(self, cfg):
        super().__init__()

        # fmt: off
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
        # Vis parameters
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT
        # fmt: on

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

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        """
        In Detectron1, loss is normalized by number of foreground samples in the batch.
        When batch size is 1 per GPU, #foreground has a large variance and
        using it lead to lower performance. Here we maintain an EMA of #foreground to
        stabilize the normalizer.
        """
        self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9

    @property
    def device(self):
        return self.pixel_mean.device


    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(results),\
            "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"].cpu().numpy()
        assert img.shape[0] == 3, "Images should have 3 channels."
        if self.input_format == "BGR":
            img = img[::-1, :, :]
        img = img.transpose(1, 2, 0)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index],
                                                 img.shape[0],
                                                 img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)


    def forward(self, batched_inputs):
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
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
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

        if self.training:

            # Panel GT
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

            # Label GT
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
            # Loss
            # label_gt_classes=label_gt_classes,
            # label_gt_anchor_deltas=label_gt_anchors_reg_deltas,
            # label_pred_class_logits=label_box_cls,
            # label_pred_anchor_deltas=label_box_delta)

            # if self.vis_period > 0:
                # storage = get_event_storage()
                # if storage.iter % self.vis_period == 0:
                    # results = self.inference(box_cls, box_delta, anchors, images.image_sizes)
                    # self.visualize_training(batched_inputs, results)

        # Inference
        else:
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
                                    gt_classes,
                                    gt_anchors_deltas,
                                    pred_class_logits,
                                    pred_anchor_deltas,
                                    num_classes):
        """
        TODO
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

        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            box_cls=pred_class_logits,
            box_delta=pred_anchor_deltas,
            num_classes=num_classes)
        # Shapes: (N x R, K) and (N x R, 4), respectively.

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

        ## PANEL LOSSES
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
                         anchors,
                         gt_instances,
                         num_classes):
        """
        Args:
            anchors (list[Boxes]):
                A list of #feature level Boxes. The Boxes contains anchors of this image on the
                specific feature level.
            targets (list[Instances]):
                A list of N `Instances`s. The i-th `Instances` contains the ground-truth
                per-instance annotations for the i-th input image.  Specify `targets` during
                training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each anchor.
                R is the total number of anchors, i.e. the sum of Hi x Wi x A for all levels.
                Anchors with an IoU with some target higher than the foreground threshold
                are assigned their corresponding label in the [0, K-1] range.
                Anchors whose IoU are below the background threshold are assigned
                the label "K". Anchors whose IoU are between the foreground and background
                thresholds are assigned a label "-1", i.e. ignore.
            gt_anchors_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth box2box transform
                targets (dx, dy, dw, dh) that map each anchor to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                anchor is labeled as foreground.
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
                  panel_box_cls,
                  panel_box_delta,
                  panel_anchors,
                  label_box_cls,
                  label_box_delta,
                  label_anchors,
                  image_sizes):
        """
        Arguments:
            box_cls, box_delta: Same as the output of :meth:`RetinaNetHead.forward`
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contain anchors of this image on the specific feature level.
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = []

        panel_box_cls = [permute_to_N_HWA_K(x, K=1) for x in panel_box_cls]
        panel_box_delta = [permute_to_N_HWA_K(x, 4) for x in panel_box_delta]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)
        label_box_cls = [permute_to_N_HWA_K(x, K=self.num_label_classes) for x in label_box_cls]
        label_box_delta = [permute_to_N_HWA_K(x, 4) for x in label_box_delta]

        for img_idx, image_size in enumerate(image_sizes):
            panel_box_cls_per_image = [box_cls_per_level[img_idx]
                                       for box_cls_per_level in panel_box_cls]
            panel_box_reg_per_image = [box_reg_per_level[img_idx]
                                       for box_reg_per_level in panel_box_delta]
            label_box_cls_per_image = [box_cls_per_level[img_idx]
                                       for box_cls_per_level in label_box_cls]
            label_box_reg_per_image = [box_reg_per_level[img_idx]
                                       for box_reg_per_level in label_box_delta]
            panel_results = self.inference_single_image(box_cls=panel_box_cls_per_image,
                                                        box_delta=panel_box_reg_per_image,
                                                        anchors=panel_anchors,
                                                        image_size=tuple(image_size),
                                                        num_classes=1)

            label_results = self.inference_single_image(box_cls=label_box_cls_per_image,
                                                        box_delta=label_box_reg_per_image,
                                                        anchors=label_anchors,
                                                        image_size=tuple(image_size),
                                                        num_classes=self.num_label_classes)

            results.append((panel_results, label_results))

        return results


    def _filter_detections(self,
                           box_cls,
                           box_delta,
                           anchors,
                           num_classes):
        """
        TODO
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


    def inference_single_image(self,
                               box_cls,
                               box_delta,
                               anchors,
                               image_size,
                               num_classes):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """

        results = Instances(image_size)
        boxes, scores, class_idxs = self._filter_detections(
            box_cls=box_cls,
            box_delta=box_delta,
            anchors=anchors,
            num_classes=num_classes)

        results.pred_boxes = Boxes(boxes)
        results.scores = scores
        results.pred_classes = class_idxs

        return results


    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.panel_fpn.size_divisibility)
        return images


class RetinaNetHead(nn.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self,
                 cfg,
                 input_shape: List[ShapeSpec],
                 num_classes,
                 num_anchors):
        super().__init__()

        in_channels = input_shape[0].channels
        num_convs = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB

        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
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


    def forward(self, features):
        """
        Args:
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
        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg
