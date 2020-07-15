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


######################################
Evaluation tool for caption splitting.
"""

from typing import Dict
import textdistance

from ..utils.figure import Figure
from ..data.figure_generators import FigureGenerator


def caption_splitting_figure_eval(figure: Figure, stat_dict: Dict[str, any]):
    """
    Evaluate caption splitting metrics on a single figure.

    Args:
        figure (Figure):            The figure on which to evaluate the caption
                                        splitting task.
        stat_dict (Dict[str, any]): A dict containing caption splitting
                                        evaluation stats It will be updated by
                                        this function.
    """

    for gt_subfigure in figure.gt_subfigures:
        if gt_subfigure.caption is not None:
            pass


def evaluate_predictions(figure_generator: FigureGenerator):
    """
    Compute the metrics from a given set of predicted sub captions.

    Args:
        figure_generator (FigureGenerator): A figure generator yielding Figure objects
                                                augmented with detected sub captions.

    Returns:
        metrics (dict): A dict containing the computed metrics.
    """

    stat_dict = {
    }

    for figure in figure_generator():

        caption_splitting_figure_eval(figure, stat_dict)
