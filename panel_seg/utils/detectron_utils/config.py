from detectron2.config import CfgNode as CN


def add_evaluation_config(cfg):
    """
    Add config for the evaluation feature.
    """
    _C = cfg

    # Name of the validation data set
    _C.DATASETS.VALIDATION = ""

    _C.VALIDATION = CN()

    # The time between each evaluation
    _C.VALIDATION.VALIDATION_PERIOD = 1000
