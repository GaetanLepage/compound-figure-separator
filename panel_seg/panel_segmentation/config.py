"""
TODO
"""

from detectron2.config import CfgNode as CN

def add_panel_seg_config(cfg):
    """
    Add config for the panel seg adapted retinanet.
    """
    _C = cfg

    # Name of the validation data set
    _C.MODEL.RETINANET.PANEL_IN_FEATURES = []
    _C.MODEL.RETINANET.LABEL_IN_FEATURES = []

    _C.MODEL.RETINANET.NUM_LABEL_CLASSES = 50

    _C.MODEL.PANEL_FPN = CN()
    _C.MODEL.PANEL_FPN.IN_FEATURES = []
    _C.MODEL.PANEL_FPN.OUT_CHANNELS = 0

    _C.MODEL.LABEL_FPN = CN()
    _C.MODEL.LABEL_FPN.IN_FEATURES = []
    _C.MODEL.LABEL_FPN.OUT_CHANNELS = 0

    _C.MODEL.PANEL_ANCHOR_GENERATOR = CN()
    _C.MODEL.PANEL_ANCHOR_GENERATOR.SIZES = []
    _C.MODEL.PANEL_ANCHOR_GENERATOR.ASPECT_RATIOS = []

    _C.MODEL.LABEL_ANCHOR_GENERATOR = CN()
    _C.MODEL.LABEL_ANCHOR_GENERATOR.SIZES = []
    _C.MODEL.LABEL_ANCHOR_GENERATOR.ASPECT_RATIOS = []
