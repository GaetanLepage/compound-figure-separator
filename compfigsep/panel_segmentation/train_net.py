#!/usr/bin/env python

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


#############################################
Panel Segmentation detection training script.

This scripts reads a given config file and runs the training or evaluation.
"""

import os
from typing import List

import torch
from torch.utils.data import DataLoader
from torch import nn

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
    HookBase)
from detectron2.evaluation import verify_results
from detectron2.data.build import build_detection_train_loader, build_detection_test_loader

from compfigsep.panel_segmentation.dataset_mapper import PanelSegDatasetMapper
from compfigsep.utils.detectron_utils import (
    LossEvalHook,
    ModelWriter
)
from compfigsep.panel_segmentation import (
    register_panel_segmentation_dataset,
    PanelSegEvaluator,
    add_panel_seg_config,
    PanelSegRetinaNet)
from compfigsep.utils.detectron_utils.config import add_validation_config


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow.

    Here, the Trainer is able to perform validation.
    """

    @classmethod
    def build_evaluator(cls,
                        cfg: CfgNode,
                        dataset_name: str,
                        output_folder: str = None) -> PanelSegEvaluator:
        """
        Builds the PanelSegEvaluator that will be called at test time.

        Args:
            cfg (CfgNode):          The global config.
            dataset_name (str):     The name of the test data set.
            output_folder (str):    The path to the folder where to store the inference results.

        Returns:
            PanelSegEvaluator:  The evaluator for testing label recognition results.
        """
        # TODO implement export functionnality
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        return PanelSegEvaluator(dataset_name)


    @classmethod
    def build_train_loader(cls,
                           cfg: CfgNode) -> DataLoader:
        """
        Instanciate the training data loader.

        Args:
            cfg (CfgNode):  The global config.

        Returns:
            a DataLoader yielding formatted training examples.
        """
        mapper = PanelSegDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg,
                                            mapper=mapper)


    @classmethod
    def build_test_loader(cls,
                          cfg: CfgNode,
                          dataset_name: str) -> DataLoader:
        """
        Instanciate the test data loader.

        Args:
            cfg (CfgNode):      The global config.
            dataset_name (str): The name of the test dataset.

        Returns:
            a DataLoader yielding formatted test examples.
        """
        mapper = PanelSegDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg,
                                           dataset_name=dataset_name,
                                           mapper=mapper)


    def build_hooks(self) -> List[HookBase]:
        """
        This method overwrites the default one from DefaultTrainer.
        It adds the `LossEvalHook` that allows evaluating results on the validation set.
        The default method builds a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]: The list of hooks to call during training.
        """
        hooks = super().build_hooks()

        # TODO remove as it can't work
        # input_example = next(iter(self.data_loader))
        # hooks.append(ModelWriter(model=self.model,
                                 # input_example=input_example,
                                 # log_dir=self.cfg.OUTPUT_DIR))

        # We add our custom validation hook
        if self.cfg.DATASETS.VALIDATION != "":
            data_set_mapper = PanelSegDatasetMapper(cfg=self.cfg,
                                                    is_train=True)
            data_loader = build_detection_test_loader(cfg=self.cfg,
                                                      dataset_name=self.cfg.DATASETS.VALIDATION,
                                                      mapper=data_set_mapper)

            loss_eval_hook = LossEvalHook(eval_period=self.cfg.VALIDATION.VALIDATION_PERIOD,
                                          model=self.model,
                                          data_loader=data_loader)
            hooks.insert(index=-1,
                         obj=loss_eval_hook)

        return hooks


    @classmethod
    def build_model(cls, cfg: CfgNode) -> nn.Module:
        """
        Instanciate and return the PanelSegRetinaNet model.

        Args:
            cfg (CfgNode):  The global config.

        Returns:
            model (nn.Module):  The PanelSegRetinaNet model.
        """
        model = PanelSegRetinaNet(cfg)
        model.to(torch.device(cfg.MODEL.DEVICE))
        logger = setup_logger(name=__name__,
                              distributed_rank=comm.get_rank())
        logger.info("Model:\n%s", model)
        return model


def setup(args: List[str]) -> CfgNode:
    """
    Create configs and perform basic setups.

    Args:
        args (List[str]):   Arguments from the command line.

    Retuns:
        cfg (CfgNode):  A config node filled with necessary options.
    """
    cfg = get_cfg()

    # Add some config options to handle validation
    add_validation_config(cfg)

    # Add some config options relative to the panel segmentation task.
    add_panel_seg_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def register_datasets(cfg: CfgNode):
    """
    Register the data sets needed for label recognition in Detectron 2's registry.

    Args:
        cfg (CfgNode): The config node filled with necessary options.
    """
    # Training
    for dataset_name in cfg.DATASETS.TRAIN:
        register_panel_segmentation_dataset(dataset_name=dataset_name)

    # Test
    for dataset_name in cfg.DATASETS.TEST:
        register_panel_segmentation_dataset(dataset_name=dataset_name)

    # Validation
    if cfg.DATASETS.VALIDATION != "":
        register_panel_segmentation_dataset(dataset_name=cfg.DATASETS.VALIDATION)


def main(args: List[str]) -> dict:
    """
    Launch training/testing for the panel splitting task on a single device.

    Args:
        args (List[str]): Arguments from the command line.

    Returns:
        If training: OrderedDict of results, if evaluation is enabled. Otherwise None.
        If test: a dict of result metrics.
    """
    cfg = setup(args)

    # Register the needed datasets
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
