# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from detectron2.structures import BoxMode, Instances, Boxes

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["PanelSegDatasetMapper"]


# def filter_empty_instances(instances, by_box=True, box_threshold=1e-5):
    # """
    # Filter out empty instances in an `Instances` object.

    # Args:
        # instances (Instances):
        # by_box (bool): whether to filter out instances with empty boxes
        # by_mask (bool): whether to filter out instances with empty masks
        # box_threshold (float): minimum width and height to be considered non-empty

    # Returns:
        # Instances: the filtered instances.
    # """
    # r = []
    # r.append(instances.gt_boxes.nonempty(threshold=box_threshold))

    # if not r:
        # return instances
    # m = r[0]
    # for x in r[1:]:
        # m = m & x
    # return instances[m]



class PanelSegDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        self.img_format = cfg.INPUT.FORMAT

        self.is_train = is_train


    @staticmethod
    def _transform_instance_annotations(annotation,
                                        transforms):
        """
        Apply transforms to box, segmentation and keypoints annotations of a single instance.

        It will use `transforms.apply_box` for the box, and
        `transforms.apply_coords` for segmentation polygons & keypoints.
        If you need anything more specially designed for each data structure,
        you'll need to implement your own version of this function or the transforms.

        Args:
            annotation (dict): dict of instance annotations for a single instance.
                It will be modified in-place.
            transforms (TransformList):
            image_size (tuple): the height, width of the transformed image
            keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

        Returns:
            dict:
                the same input dict with fields "bbox", "segmentation", "keypoints"
                transformed according to `transforms`.
                The "bbox_mode" field will be set to XYXY_ABS.
        """
        panel_bbox = BoxMode.convert(box=annotation["panel_bbox"],
                                     from_mode=annotation["bbox_mode"],
                                     to_mode=BoxMode.XYXY_ABS)
        label_bbox = BoxMode.convert(box=annotation["label_bbox"],
                                     from_mode=annotation["bbox_mode"],
                                     to_mode=BoxMode.XYXY_ABS)
        # Note that bbox is 1d (per-instance bounding box)
        annotation["panel_bbox"] = transforms.apply_box([panel_bbox])[0]
        annotation["label_bbox"] = transforms.apply_box([label_bbox])[0]
        annotation["bbox_mode"] = BoxMode.XYXY_ABS

        return annotation


    @staticmethod
    def annotations_to_instances(annos, image_size):
        """
        Create an :class:`Instances` object used by the models,
        from instance annotations in the dataset dict.

        Args:
            annos (list[dict]): a list of instance annotations in one image, each
                element for one instance.
            image_size (tuple): height, width

        Returns:
            Instances:
                It will contain fields "gt_boxes", "gt_classes",
                "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
                This is the format that builtin models expect.
        """
        target = Instances(image_size)

        # Panels
        panel_boxes = [BoxMode.convert(box=obj["panel_bbox"],
                                       from_mode=obj["bbox_mode"],
                                       to_mode=BoxMode.XYXY_ABS)
                       for obj in annos]
        panel_boxes = target.panel_gt_boxes = Boxes(panel_boxes)
        panel_boxes.clip(image_size)

        # Only one class (panel)
        panel_classes = [0 for obj in annos]
        panel_classes = torch.tensor(panel_classes, dtype=torch.int64)
        target.panel_gt_classes = panel_classes


        # Labels
        label_boxes = [BoxMode.convert(box=obj["label_bbox"],
                                       from_mode=obj["bbox_mode"],
                                       to_mode=BoxMode.XYXY_ABS)
                       for obj in annos]
        label_boxes = target.label_gt_boxes = Boxes(label_boxes)
        label_boxes.clip(image_size)

        label_classes = [obj["category_id"] for obj in annos]
        label_classes = torch.tensor(label_classes, dtype=torch.int64)
        target.label_gt_classes = label_classes

        return target


    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                transform_gens=([self.crop_gen] if self.crop_gen else []) + self.tfm_gens,
                img=image)

        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        # if self.load_proposals:
            # utils.transform_proposals(dataset_dict,
                                      # image_shape,
                                      # transforms,
                                      # self.min_box_side_len,
                                      # self.proposal_topk)

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Implement additional transformations if you have other types of data
            annos = [
                self._transform_instance_annotations(obj,
                                                     transforms)

                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = self.annotations_to_instances(annos,
                                                      image_shape)
            # Create a tight bounding box from masks, useful when image is cropped
            # if self.crop_gen and instances.has("gt_masks"):
                # instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # dataset_dict["instances"] = utils.filter_empty_instances(instances)

            # TODO check if we have to adapt this for handling panel_boxes and label_boxes
            # (see method above)
            # dataset_dict["instances"] = utils.filter_empty_instances(instances)
            dataset_dict["instances"] = instances

        return dataset_dict
