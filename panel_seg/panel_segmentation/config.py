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


########################################################################################
TODO
"""

from detectron2.config import CfgNode

def add_panel_seg_config(cfg: CfgNode):
    """
    Add config for the panel seg adapted retinanet.

    Args:
        cfg (CfgNode):  The config node that will be extended.
    """
    _C = cfg

    # Name of the validation data set
    _C.MODEL.RETINANET.PANEL_IN_FEATURES = []
    _C.MODEL.RETINANET.LABEL_IN_FEATURES = []

    _C.MODEL.RETINANET.NUM_LABEL_CLASSES = 50

    _C.MODEL.PANEL_FPN = CfgNode()
    _C.MODEL.PANEL_FPN.IN_FEATURES = []
    _C.MODEL.PANEL_FPN.OUT_CHANNELS = 0

    _C.MODEL.LABEL_FPN = CfgNode()
    _C.MODEL.LABEL_FPN.IN_FEATURES = []
    _C.MODEL.LABEL_FPN.OUT_CHANNELS = 0

    _C.MODEL.PANEL_ANCHOR_GENERATOR = CfgNode()
    _C.MODEL.PANEL_ANCHOR_GENERATOR.SIZES = []
    _C.MODEL.PANEL_ANCHOR_GENERATOR.ASPECT_RATIOS = []

    _C.MODEL.LABEL_ANCHOR_GENERATOR = CfgNode()
    _C.MODEL.LABEL_ANCHOR_GENERATOR.SIZES = []
    _C.MODEL.LABEL_ANCHOR_GENERATOR.ASPECT_RATIOS = []
