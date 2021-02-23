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


#################################################################
Function to filter out unlikely panels.
"""

from typing import List, Dict, Set

from ..utils.figure.panel import DetectedPanel
from ..utils import box


def filter_panels(panel_list: List[DetectedPanel],
                  overlap_threshold: float = 0.8) -> List[DetectedPanel]:
    """
    TODO

    Args:
        panel_list (List[DetectedPanel]):   A list of detected panels.
    """

    kept_panels: List[DetectedPanel] = []

    for index_1, panel_1 in enumerate(panel_list):

        for index_2, panel_2 in enumerate(kept_panels):

            if index_1 == index_2:
                continue

            overlap: float = box.intersection_area(box_1=panel_1.box,
                                                   box_2=panel_2.box) / box.area(panel_1.box)

            if overlap < overlap_threshold:
                kept_panels.pop(index_1)
                break


    return kept_panels
