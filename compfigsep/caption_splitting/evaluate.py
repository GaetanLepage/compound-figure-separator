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

from typing import Dict, Any
from pprint import pprint

import textdistance # type: ignore

from ..utils.figure import Figure
from ..data.figure_generators import FigureGenerator


def caption_splitting_figure_eval(figure: Figure,
                                  stat_dict: Dict[str, Any]) -> None:
    """
    Evaluate caption splitting metrics on a single figure.

    Args:
        figure (Figure):            The figure on which to evaluate the caption
                                        splitting task.
        stat_dict (Dict[str, any]): A dict containing caption splitting
                                        evaluation stats It will be updated by
                                        this function.
    """
    # Score of this figure
    figure_score: float = 0

    # Number of ground truth labels. i.e. the normalizer for the figure score.
    num_gt_labels: int = 0

    # Get the dictionnary of detected subcaptions (if it exists).
    detected_subcaptions: Dict[str, str] = figure.detected_subcaptions\
                                           if hasattr(figure, 'detected_subcaptions')\
                                           else {}

    if not hasattr(figure, 'gt_subfigures'):
        return

    # Case where this is not a compound figure.
    if len(figure.gt_subfigures) == 1 and '_' in figure.detected_subcaptions:
        num_gt_labels = 1
        figure_score = textdistance.levenshtein.normalized_similarity(
            figure.gt_subfigures[0],
            figure.detected_subcaptions['_'])

    # Case where multiple labels where detected.
    else:
        for gt_subfigure in figure.gt_subfigures:

            if gt_subfigure.caption is None:
                continue

            if gt_subfigure.label is None or gt_subfigure.label.text is None:
                continue

            # Increase the number of ground truth labels.
            num_gt_labels += 1

            if gt_subfigure.label.text in detected_subcaptions:

                # Get the ground truth subcaption.
                gt_subcaption: str = gt_subfigure.caption

                # Get the corresponding detected subcaption.
                detected_subcaption: str = \
                    detected_subcaptions[gt_subfigure.label.text]

                # Compute the Levenshtein distance between the GT and the detection.
                figure_score += textdistance.levenshtein.normalized_similarity(
                    gt_subcaption,
                    detected_subcaption)

    # If the ground truth figure was not annotated,
    if num_gt_labels > 0:
        stat_dict['num_captions'] += 1
        stat_dict['levenshtein_metric'] += figure_score / num_gt_labels


def caption_splitting_metrics(stat_dict: Dict[str, Any]) -> float:
    """
    Compute the metrics for the caption splitting task.

    Args:
        stat_dict (Dict[str, any]): A dict containing the stats gathered while looping over
                                        detections.

    Returns:
        levenshtein_metric (float):  The averaged levenshtein metric.
    """

    return stat_dict['levenshtein_metric'] / stat_dict['num_captions']


def evaluate_detections(figure_generator: FigureGenerator) -> Dict[str, float]:
    """
    Compute the metrics from a given set of predicted sub captions.

    Args:
        figure_generator (FigureGenerator): A figure generator yielding Figure objects
                                                augmented with detected sub captions.

    Returns:
        metrics (Dict[str, any]): A dict containing the computed metrics.
    """

    stat_dict: Dict[str, Any] = {
        'num_captions': 0,
        'levenshtein_metric': 0
    }

    for figure in figure_generator():

        caption_splitting_figure_eval(figure, stat_dict)

    lavenstein_metric: float = caption_splitting_metrics(stat_dict=stat_dict)

    metrics: Dict[str, float] = {
        'levenstein_metric': lavenstein_metric
    }

    pprint(metrics)

    return metrics
