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
from panel_seg.utils.loss_eval_hook import LossEvalHook
from panel_seg.panel_split.model.panel_split_evaluator import PanelSplitEvaluator
from panel_seg.panel_split.model.config import add_evaluation_config


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """

                 # evaluation_period: int,
                 # evaluation_dataset_name: str):
    def __init__(self,
                 cfg: CfgNode):
        """
        TODO doc

        Args:
            cfg (CfgNode): TODO
            evaluation_period (int): TODO
            evaluation_dataset_name (str): TODO
        """
        self._validation_period = cfg.VALIDATION.VALIDATION_PERIOD
        # self._validation_period = evaluation_period
        self._validation_dataset_name = cfg.DATASETS.VALIDATION
        # self._validation_dataset_name = evaluation_dataset_name
        super().__init__(cfg)


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
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
        TODO
        """
        hooks = super().build_hooks()

        hooks.insert(-1,
                     LossEvalHook(eval_period=self._validation_period,
                                  model=self.model,
                                  data_loader=build_detection_test_loader(
                                      cfg=self.cfg,
                                      dataset_name=self._validation_dataset_name,
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

    register_image_clef_datasets()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model,
                              save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                      resume=args.resume)
        res = Trainer.test(cfg, model)
        # if cfg.TEST.AUG.ENABLED:
            # res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    # If you'd like to do anything fancier than the standard training logic,
    # consider writing your own training loop or subclassing the trainer.
    trainer = Trainer(cfg)
                      # args.evaluation_period,
                      # args.evaluation_dataset_name)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--eval-period",
                        help="The number of iterations between two validations.",
                        type=int,
                        default=5000)

    args = parser.parse_args()
    print("Command Line Args:", args)
    # load_datasets(args.dataset)
    launch(main,
           args.num_gpus,
           num_machines=args.num_machines,
           machine_rank=args.machine_rank,
           dist_url=args.dist_url,
           args=(args,))
