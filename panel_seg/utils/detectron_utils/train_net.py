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

from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results
from detectron2.data.build import build_detection_test_loader
from detectron2.data.dataset_mapper import DatasetMapper

from panel_seg.panel_split.model.load_panel_split_datasets import register_image_clef_datasets
from panel_seg.utils.detectron_utils.loss_eval_hook import LossEvalHook
from panel_seg.panel_split.model.panel_split_evaluator import PanelSplitEvaluator
from panel_seg.utils.detectron_utils.config import add_evaluation_config


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow.
    Here, the Trainer is able to perform validation.
    """

    @classmethod
    def build_evaluator(cls,
                        cfg,
                        dataset_name,
                        output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        return PanelSplitEvaluator(dataset_name, output_folder)


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
        hooks.insert(-1,
                     LossEvalHook(eval_period=self.cfg.VALIDATION.VALIDATION_PERIOD,
                                  model=self.model,
                                  data_loader=build_detection_test_loader(
                                      cfg=self.cfg,
                                      dataset_name=self.cfg.DATASETS.VALIDATION,
                                      mapper=DatasetMapper(cfg=self.cfg,
                                                           is_train=True))))
        return hooks



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_evaluation_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    # TODO Clean dataset ingest
    register_image_clef_datasets()

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
