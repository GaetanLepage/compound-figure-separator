#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
"""

import os
import logging

import torch

from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results
from detectron2.data.build import build_detection_train_loader, build_detection_test_loader

from panel_seg.panel_segmentation.dataset_mapper import PanelSegDatasetMapper
from panel_seg.panel_segmentation.load_panel_segmentation_datasets import register_panel_segmentation_dataset
from panel_seg.utils.detectron_utils.loss_eval_hook import LossEvalHook
from panel_seg.panel_segmentation.panel_seg_evaluator import PanelSegEvaluator
from panel_seg.utils.detectron_utils.config import add_validation_config
from panel_seg.panel_segmentation.config import add_panel_seg_config

from panel_seg.panel_segmentation.panel_seg_retinanet import PanelSegRetinaNet


class Trainer(DefaultTrainer):
    """
    TODO : do une train_net.py per task... or find a better way to split the work

    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow.
    Here, the Trainer is able to perform validation.
    """

    # def __init__(self, cfg):
        # """
        # TODO
        # """

        # # TODO maybe overwrite to add dataset mapper as attribute

        # # TODO maybe overwrite to add dataset mapper as attribute
        # super().__init__(cfg)


    @classmethod
    def build_evaluator(cls,
                        cfg,
                        dataset_name,
                        output_folder=None):
        """
        TODO
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        return PanelSegEvaluator(dataset_name)


    @classmethod
    def build_train_loader(cls, cfg):
        """
        TODO
        """
        mapper = PanelSegDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg,
                                            mapper=mapper)


    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        TODO
        """
        mapper = PanelSegDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg,
                                           dataset_name=dataset_name,
                                           mapper=mapper)


    def build_hooks(self):
        """
        This method overwrites the default one from DefaultTrainer.
        It adds the `LossEvalHook` that allows
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        hooks = super().build_hooks()

        # We add our custom validation hook
        if self.cfg.DATASETS.VALIDATION != "":
            hooks.insert(-1,
                         LossEvalHook(eval_period=self.cfg.VALIDATION.VALIDATION_PERIOD,
                                      model=self.model,
                                      data_loader=build_detection_test_loader(
                                          cfg=self.cfg,
                                          dataset_name=self.cfg.DATASETS.VALIDATION,
                                          mapper=PanelSegDatasetMapper(cfg=self.cfg,
                                                                       is_train=True))))

        return hooks


    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = PanelSegRetinaNet(cfg)
        model.to(torch.device(cfg.MODEL.DEVICE))
        logger = setup_logger(name=__name__,
                              distributed_rank=comm.get_rank())
        logger.info("Model:\n{}".format(model))
        return model


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_validation_config(cfg)
    add_panel_seg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def register_datasets(cfg):
    """
    TODO
    """
    for dataset_name in cfg.DATASETS.TRAIN:
        register_panel_segmentation_dataset(dataset_name=dataset_name)

    for dataset_name in cfg.DATASETS.TEST:
        register_panel_segmentation_dataset(dataset_name=dataset_name)

    if cfg.DATASETS.VALIDATION != "":
        register_panel_segmentation_dataset(dataset_name=cfg.DATASETS.VALIDATION)


def main(args):
    cfg = setup(args)

    register_datasets(cfg)

    # Inference only (testing)
    if args.eval_only:

        # Load the model
        model = Trainer.build_model(cfg)

        # Load the latest weights
        DetectionCheckpointer(model,
                              save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                      resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    # Training
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()

    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(main,
           args.num_gpus,
           num_machines=args.num_machines,
           machine_rank=args.machine_rank,
           dist_url=args.dist_url,
           args=(args,))
