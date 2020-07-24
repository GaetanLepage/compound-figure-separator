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


#######################################################################################
Additional config options to handle the evaluation of a validation set during training.
"""

from detectron2.config import CfgNode # type: ignore

def add_validation_config(cfg: CfgNode) -> None:
    """
    Add config for the evaluation feature.

    Args:
        cfg (CfgNode):  The config node that will be extended.
    """
    # Name of the validation data set
    cfg.DATASETS.VALIDATION = ""

    cfg.VALIDATION = CfgNode()

    # The time between each evaluation
    cfg.VALIDATION.VALIDATION_PERIOD = 0
