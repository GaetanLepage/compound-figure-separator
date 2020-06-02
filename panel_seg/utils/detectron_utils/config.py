"""
TODO
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
