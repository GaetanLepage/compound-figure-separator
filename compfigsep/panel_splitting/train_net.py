#!/usr/bin/env python3

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


###########################################################################
Panel Splitting detection training script.

This scripts reads a given config file and runs the training or evaluation.
"""

from argparse import ArgumentParser, Namespace
from typing import List

from torch.utils.data import DataLoader
from torch import nn

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.engine import (DefaultTrainer,
                               default_argument_parser,
                               default_setup,
                               launch,
                               HookBase)
from detectron2.evaluation import verify_results
from detectron2.data.build import build_detection_test_loader
from detectron2.data.dataset_mapper import DatasetMapper

from compfigsep.panel_splitting import register_panel_splitting_dataset, PanelSplitEvaluator
from compfigsep.utils.detectron_utils import LossEvalHook, add_validation_config


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow.

    Here, the Trainer is able to perform validation.
    """

    @classmethod
    def build_evaluator(cls,
                        cfg: CfgNode,
                        dataset_name: str) -> PanelSplitEvaluator:
        """
        Builds the PanelSplitEvaluator that will be called at test time.

        Args:
            cfg (CfgNode):          The config node filled with necessary options.
            dataset_name (str):     The name of the test data set.

        Returns:
            PanelSplitEvaluator: The evaluator for testing label recognition results.
        """
        return PanelSplitEvaluator(dataset_name=dataset_name,
                                   export=True,
                                   export_dir=cfg.OUTPUT_DIR)


    def build_hooks(self) -> List[HookBase]:
        """
        This method overwrites the default one from DefaultTrainer.
        It adds the `LossEvalHook` that allows
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            List[HookBase]: The augmented list of hooks.
        """
        hooks: List[HookBase] = super().build_hooks()

        # We add our custom validation hook
        if self.cfg.DATASETS.VALIDATION != "":
            data_set_mapper: DatasetMapper = DatasetMapper.from_config(cfg=self.cfg,
                                                                       is_train=True)

            data_loader: DataLoader = build_detection_test_loader(cfg=self.cfg,
                                                      dataset_name=self.cfg.DATASETS.VALIDATION,
                                                      mapper=data_set_mapper)

            loss_eval_hook: LossEvalHook = LossEvalHook(
                eval_period=self.cfg.VALIDATION.VALIDATION_PERIOD,
                model=self.model,
                data_loader=data_loader)

            hooks.insert(-1, loss_eval_hook)

        return hooks



def setup(parsed_args: Namespace) -> CfgNode:
    """
    Create configs and perform basic setups.

    Args:
        args (Namespace):   Arguments from the command line.

    Retuns:
        cfg (CfgNode):  A config node filled with necessary options.
    """
    cfg: CfgNode = get_cfg()

    # Add some config options to handle validation
    add_validation_config(cfg)

    cfg.merge_from_file(parsed_args.config_file)
    cfg.merge_from_list(parsed_args.opts)
    cfg.freeze()
    default_setup(cfg, parsed_args)
    return cfg


def register_datasets(cfg: CfgNode):
    """
    Register the data sets needed for panel splitting in Detectron 2's registry.

    Args:
        cfg (CfgNode):  The config node filled with necessary options.
    """
    # Training
    for dataset_name in cfg.DATASETS.TRAIN:
        register_panel_splitting_dataset(dataset_name=dataset_name)

    # Test
    for dataset_name in cfg.DATASETS.TEST:
        register_panel_splitting_dataset(dataset_name=dataset_name)

    # Validation
    if cfg.DATASETS.VALIDATION != "":
        register_panel_splitting_dataset(dataset_name=cfg.DATASETS.VALIDATION)


def main(parsed_args: Namespace) -> dict:
    """
    Launch training/testing for the panel splitting task on a single device.

    Args:
        parsed_args (Namespace):    Arguments from the command line.

    Returns:
        If training:    OrderedDict of results, if evaluation is enabled. Otherwise None.
        If test:    A dict of result metrics.
    """
    cfg: CfgNode = setup(parsed_args)

    # Register the needed datasets
    register_datasets(cfg)

    # Inference only (testing)
    if parsed_args.eval_only:

        # Load the model
        model: nn.Module = Trainer.build_model(cfg)

        # Load the latest weights
        DetectionCheckpointer(model,
                              save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                      resume=parsed_args.resume)
        res: dict = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    # Training
    trainer: Trainer = Trainer(cfg)
    trainer.resume_or_load(resume=parsed_args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser: ArgumentParser = default_argument_parser()

    parsed_args: Namespace = parser.parse_args()

    launch(main,
           parsed_args.num_gpus,
           num_machines=parsed_args.num_machines,
           machine_rank=parsed_args.machine_rank,
           dist_url=parsed_args.dist_url,
           args=(parsed_args,))
