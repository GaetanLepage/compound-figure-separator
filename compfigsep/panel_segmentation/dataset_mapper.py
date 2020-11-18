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


###################################################################################
This file contains the mapping that is applied to panel segmentation dataset dicts.

Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""

from typing import List

import copy
import logging
import numpy as np # type: ignore
import torch

from detectron2.data import detection_utils as utils # type: ignore
from detectron2.data import transforms as T

from detectron2.structures import BoxMode, Instances, Boxes # type: ignore
from detectron2.config import CfgNode # type: ignore


__all__ = ["PanelSegDatasetMapper"]


class PanelSegDatasetMapper:
    """
    A callable which takes a dataset dict in Panel Segmentation Dataset format,
    and map it into a format used by the model.

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`

    Attributes:
        crop_gen (RandomCrop):          The transformation that will be applied to samples.
        tfm_gen (list[TransformGen]):   A list of :class:`TransformGen` from config.
        img_format (str):               The image format (ex: 'BGR').
        is_train (bool):                Whether we are currently training or testing.
    """

    def __init__(self,
                 cfg: CfgNode,
                 is_train: bool = True):
        """
        Init method for class PanelSegDatasetMapper.

        Args:
            cfg (CfgNode):      The config node filled with necessary options.
            is_train (bool):    Whether we are currently training or testing.
        """

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: %s", str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        self.img_format = cfg.INPUT.FORMAT

        self.is_train = is_train


    @staticmethod
    def _annotations_to_instances(panel_annos: List[dict],
                                  label_annos: List[dict],
                                  image_size: tuple) -> Instances:
        """
        Create an :class:`Instances` object used by the models,
        from instance annotations in the dataset dict.

        Args:
            annos (List[dict]): a list of instance annotations in one image, each
                                    element for one instance.
            image_size (tuple): height, width

        Returns:
            Instances: It will contain fields "gt_boxes", "gt_classes", if they can be obtained
                            from `annos`. This is the format that builtin models expect.
        """
        # Panels

        # Create an `Instances` object for panels
        panel_instances = Instances(image_size)
        # TODO remove
        # print("panel_instances:", panel_instances.get_fields())

        panel_boxes = [BoxMode.convert(box=obj['bbox'],
                                       from_mode=obj['bbox_mode'],
                                       to_mode=BoxMode.XYXY_ABS)
                       for obj in panel_annos]
        panel_boxes = panel_instances.gt_boxes = Boxes(panel_boxes)
        panel_boxes.clip(image_size)

        # Only one class (panel)
        panel_classes = [0 for obj in panel_annos]
        panel_classes = torch.tensor(panel_classes, dtype=torch.int64)
        panel_instances.gt_classes = panel_classes


        # Labels (also handle case where there are no labels)
        label_boxes = [BoxMode.convert(box=obj['bbox'],
                                       from_mode=obj['bbox_mode'],
                                       to_mode=BoxMode.XYXY_ABS)
                       for obj in label_annos]

        # Create an `Instances` object for labels
        label_instances = Instances(image_size)
        # if len(label_boxes) > 0:

        label_boxes = label_instances.gt_boxes = Boxes(label_boxes)
        label_boxes.clip(image_size)

        label_classes = [obj['label'] for obj in label_annos]
        label_classes = torch.tensor(label_classes, dtype=torch.int64)
        label_instances.gt_classes = label_classes

        assert len(label_boxes) == len(label_classes),\
            f"There are {len(label_boxes)} boxes but {len(label_classes)} labels."

        return panel_instances, label_instances


    def __call__(self, dataset_dict: dict) -> dict:
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

        # TODO simplify this
        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                transform_gens=([self.crop_gen]
                                if self.crop_gen
                                else []) + self.tfm_gens,
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
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop('annotations', None)
            return dataset_dict

        panel_annos = []
        label_annos = []
        for obj in dataset_dict.pop('annotations'):
            panel_obj = {
                'bbox': obj['panel_bbox'],
                'bbox_mode': obj['bbox_mode']
            }
            panel_annos.append(utils.transform_instance_annotations(annotation=panel_obj,
                                                                    transforms=transforms,
                                                                    image_size=image_shape))

            if 'label' in obj:
                if 'label_bbox' in obj:
                    label_obj = {
                        'bbox': obj['label_bbox'],
                        'bbox_mode': obj['bbox_mode'],
                        'label': obj['label']
                    }

                    label_annos.append(
                        utils.transform_instance_annotations(annotation=label_obj,
                                                             transforms=transforms,
                                                             image_size=image_shape))
                else:
                    logging.error("Error with annotation: %s has a 'label' field"\
                                  " but no corresponding 'label_bbox' field.",
                                  obj)

            elif 'label_bbox' in obj:
                logging.error("Inconsistent label annotation:"\
                              " obj['label']=%s and obj['label_bbox']=%s",
                              obj['label'],
                              obj['label_bbox'])


        panel_instances, label_instances = self._annotations_to_instances(panel_annos=panel_annos,
                                                                          label_annos=label_annos,
                                                                          image_size=image_shape)

        # dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # TODO check if we have to adapt this for handling panel_boxes and label_boxes
        # (see method above)
        # dataset_dict["instances"] = utils.filter_empty_instances(instances)
        dataset_dict['panel_instances'] = panel_instances
        dataset_dict['label_instances'] = label_instances

        return dataset_dict
