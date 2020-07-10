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


#################################################
Post-processing tools to filter label detections.
"""

from ..data.figure_generators import FigureGenerator
from ..utils.figure import Figure


def filter_labels(figure_generator: FigureGenerator):
    """
    Filter out false positive label detections.

    Args:
        figure_generator (FigureGenerator): A generator of Figure objects.

    Returns:
        TODO (TODO): TODO (do we yield figures or do we create a FigureGenerator on the fly, do we return a list of Figure objects...)
    """
