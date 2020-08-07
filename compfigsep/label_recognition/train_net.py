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

Collaborators:  Niccolò Marini (niccolo.marini@hevs.ch)
                Stefano Marchesin (stefano.marchesin@unipd.it)


###########################################################################
Label Recognition detection training script.

This scripts reads a given config file and runs the training or evaluation.
"""

from argparse import Namespace, ArgumentParser
from typing import List, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

import detectron2.utils.comm as comm # type: ignore
from detectron2.utils.logger import setup_logger # type: ignore
from detectron2.checkpoint import DetectionCheckpointer # type: ignore
from detectron2.config import CfgNode, get_cfg # type: ignore
from detectron2.engine import (DefaultTrainer, # type: ignore
                               default_argument_parser, # type: ignore
                               default_setup, # type: ignore
                               launch, # type: ignore
                               HookBase) # type: ignore
from detectron2.evaluation import verify_results # type: ignore
from detectron2.data.build import build_detection_test_loader # type: ignore
from detectron2.data.dataset_mapper import DatasetMapper # type: ignore

from compfigsep.label_recognition import (register_label_recognition_dataset, # type: ignore
                                          LabelRecogEvaluator, # type: ignore
                                          LabelRecogRetinaNet) # type: ignore
from compfigsep.utils.detectron_utils import LossEvalHook, add_validation_config # type: ignore


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow.

    Here, the Trainer is able to perform validation.
    """

    @classmethod
    def build_evaluator(cls,
                        cfg: CfgNode,
                        dataset_name: str) -> LabelRecogEvaluator:
        """
        Builds the LabelRecogEvaluator that will be called at test time.

        Args:
            cfg (CfgNode):          The config node filled with necessary options.
            dataset_name (str):     The name of the test data set.

        Returns:
            LabelRecogEvaluator: The evaluator for testing label recognition results.
        """
        return LabelRecogEvaluator(dataset_name=dataset_name,
                                   export=True,
                                   export_dir=cfg.OUTPUT_DIR)


    def build_hooks(self) -> List[HookBase]:
        """
        This method overwrites the default one from DefaultTrainer.
        It adds (if necessary) the `LossEvalHook` that allows evaluating the loss on the
        validation set.

        Returns:
            List[HookBase]: The augmented list of hooks.
        """
        # Build a list of default hooks, including timing, evaluation,
        # checkpointing, lr scheduling, precise BN, writing events.
        hooks = super().build_hooks()

        # We add our custom validation hook
        if self.cfg.DATASETS.VALIDATION != "":
            data_set_mapper: DatasetMapper = DatasetMapper(cfg=self.cfg,
                                                           is_train=True)

            data_loader: DataLoader = build_detection_test_loader(
                cfg=self.cfg,
                dataset_name=self.cfg.DATASETS.VALIDATION,
                mapper=data_set_mapper)

            loss_eval_hook: LossEvalHook = LossEvalHook(
                eval_period=self.cfg.VALIDATION.VALIDATION_PERIOD,
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
        model = LabelRecogRetinaNet(cfg)
        model.to(torch.device(cfg.MODEL.DEVICE))
        logger = setup_logger(name=__name__,
                              distributed_rank=comm.get_rank())
        logger.info("Model:\n%s", model)
        return model


def setup(parsed_args: Namespace) -> CfgNode:
    """
    Create configs and perform basic setups.

    Args:
        args (List[str]): Arguments from the command line.

    Retuns:
        cfg (CfgNode): A config node filled with necessary options.
    """
    cfg = get_cfg()

    # Add some config options to handle validation
    add_validation_config(cfg)

    cfg.merge_from_file(parsed_args.config_file)
    cfg.merge_from_list(parsed_args.opts)
    cfg.freeze()
    default_setup(cfg, parsed_args)

    return cfg


def register_datasets(cfg: CfgNode):
    """
    Register the data sets needed for label recognition in Detectron 2's registry.

    Args:
        cfg (CfgNode): The config node filled with necessary options.
    """
    # Training
    for dataset_name in cfg.DATASETS.TRAIN:
        register_label_recognition_dataset(dataset_name=dataset_name)

    # Test
    for dataset_name in cfg.DATASETS.TEST:
        register_label_recognition_dataset(dataset_name=dataset_name)

    # Validation
    if cfg.DATASETS.VALIDATION != "":
        register_label_recognition_dataset(dataset_name=cfg.DATASETS.VALIDATION)


def main(parsed_args: Namespace) -> Dict:
    """
    Launch training/testing for the label recognition task on a single device.

    Args:
        args (Namespace):   Arguments from the command line.

    Returns:
        If training: OrderedDict of results, if evaluation is enabled. Otherwise None.
        If test: a dict of result metrics.
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
        res: Dict = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg=cfg,
                           results=res)
        return res

    # Training
    trainer: Trainer = Trainer(cfg)
    trainer.resume_or_load(resume=parsed_args.resume)
    return trainer.train()


if __name__ == "__main__":
    PARSER: ArgumentParser = default_argument_parser()

    PARSED_ARGS: Namespace = PARSER.parse_args()
    print("Command Line Args:", PARSED_ARGS)

    launch(main,
           PARSED_ARGS.num_gpus,
           num_machines=PARSED_ARGS.num_machines,
           machine_rank=PARSED_ARGS.machine_rank,
           dist_url=PARSED_ARGS.dist_url,
           args=(PARSED_ARGS,))
