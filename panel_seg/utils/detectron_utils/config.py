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


#######################################################################################
Additional config options to handle the evaluation of a validation set during training.
"""

from detectron2.config import CfgNode

def add_validation_config(cfg: CfgNode):
    """
    Add config for the evaluation feature.

    Args:
        cfg (CfgNode):  The config node that will be extended.
    """
    _C = cfg

    # Name of the validation data set
    _C.DATASETS.VALIDATION = ""

    _C.VALIDATION = CfgNode()

    # The time between each evaluation
    _C.VALIDATION.VALIDATION_PERIOD = 0
