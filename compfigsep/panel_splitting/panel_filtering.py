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

from ..utils.figure.panel import DetectedPanel
from ..utils import box


def filter_panels(
        panel_list: list[DetectedPanel],
        overlap_threshold: float = 0.2
) -> list[DetectedPanel]:
    """
    TODO

    Args:
        panel_list (list[DetectedPanel]):   A list of detected panels.

    Returns:
        filtered_panels (list[DetectedPanel]):  The list of filtered panels.
    """

    kept_panels: list[DetectedPanel] = panel_list.copy()

    mask: list[bool] = [True for _ in range(len(panel_list))]

    # Cycle through the detections to see if we discard them.
    for index_1, panel_1 in enumerate(panel_list):

        # Find an eventual
        for index_2, panel_2 in enumerate(kept_panels):

            if index_1 == index_2:
                continue

            # If the panel was already discarded
            if not mask[index_1]:
                continue

            overlap = box.intersection_area(
                box_1=panel_1.box,
                box_2=panel_2.box
            ) / box.area(panel_1.box)

            if overlap > overlap_threshold:
                mask[index_1] = False
                break

    return [
        panel for index, panel
        in enumerate(panel_list)
        if mask[index]
    ]
