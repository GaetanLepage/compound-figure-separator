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


#######################################################
Additional configuration options for PanelSegRetinaNet.
"""

from detectron2.config import CfgNode # type: ignore

def add_panel_seg_config(cfg: CfgNode):
    """
    Add config for the panel seg adapted retinanet.

    Args:
        cfg (CfgNode):  The config node that will be extended.
    """
    # Name of the validation data set
    cfg.MODEL.RETINANET.PANEL_IN_FEATURES = []
    cfg.MODEL.RETINANET.LABEL_IN_FEATURES = []

    cfg.MODEL.RETINANET.NUM_LABEL_CLASSES = 50

    cfg.MODEL.PANEL_FPN = CfgNode()
    cfg.MODEL.PANEL_FPN.IN_FEATURES = []
    cfg.MODEL.PANEL_FPN.OUT_CHANNELS = 0

    cfg.MODEL.LABEL_FPN = CfgNode()
    cfg.MODEL.LABEL_FPN.IN_FEATURES = []
    cfg.MODEL.LABEL_FPN.OUT_CHANNELS = 0

    cfg.MODEL.PANEL_ANCHOR_GENERATOR = CfgNode()
    cfg.MODEL.PANEL_ANCHOR_GENERATOR.SIZES = []
    cfg.MODEL.PANEL_ANCHOR_GENERATOR.ASPECT_RATIOS = []

    cfg.MODEL.LABEL_ANCHOR_GENERATOR = CfgNode()
    cfg.MODEL.LABEL_ANCHOR_GENERATOR.SIZES = []
    cfg.MODEL.LABEL_ANCHOR_GENERATOR.ASPECT_RATIOS = []
