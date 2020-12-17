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


######################################
Evaluation tool for caption splitting.
"""

from typing import Dict, Any, Optional, List
from pprint import pprint

from joblib import Parallel, delayed

import textdistance

from ..utils.figure import Figure
from ..data.figure_generators import FigureGenerator

CaptionSplittingFigureResult = Optional[float]

def caption_splitting_figure_eval(figure: Figure) -> CaptionSplittingFigureResult:
    """
    Evaluate caption splitting metrics on a single figure.

    Args:
        figure (Figure):    The figure on which to evaluate the caption
                                splitting task.

    Return:
        score (CaptionSplittingFigureResult):   If the figure is properly annotated, the
                                                    Levenshtein metric for this figure is
                                                    returned.
                                                Else, None is returned.
    """
    # Score of this figure
    figure_score: float = 0

    # Number of ground truth labels. i.e. the normalizer for the figure score.
    num_gt_labels: int = 0

    # Get the dictionnary of detected subcaptions (if it exists).
    detected_subcaptions: Dict[str, str] = figure.detected_subcaptions \
                                           if hasattr(figure, 'detected_subcaptions')\
                                           else {}

    # If the figure was not annotated, return None.
    if not hasattr(figure, 'gt_subfigures'):
        return None

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

    if num_gt_labels > 0:
        return figure_score / num_gt_labels

    # If the ground truth figure was not annotated (no GT labels), return None.
    return None


def caption_splitting_metrics(results: List[CaptionSplittingFigureResult]) -> float:
    """
    Compute the metrics for the caption splitting task.

    Args:
        results (List[CaptionSplittingFigureResult]):   TODO.

    Returns:
        levenshtein_metric (float):  The averaged levenshtein metric.
    """
    # Filter out the figures for which no score was computed (missing annotations for e.g.).
    valid_results: List[float] = [res
                                  for res in results
                                  if res is not None]

    num_captions: int = len(valid_results)
    levenshtein_metric: float = sum(valid_results)

    return levenshtein_metric / num_captions


def evaluate_detections(figure_generator: FigureGenerator) -> Dict[str, float]:
    """
    Compute the metrics from a given set of predicted sub captions.

    Args:
        figure_generator (FigureGenerator): A figure generator yielding Figure objects
                                                augmented with detected sub captions.

    Returns:
        metrics (Dict[str, any]): A dict containing the computed metrics.
    """

    # Parallel computing to speed up the evaluation
    with Parallel(n_jobs=-1) as parallel:

        results: List[CaptionSplittingFigureResult] = parallel(
            [delayed(caption_splitting_figure_eval)(figure)
             for figure in figure_generator()])

    lavenstein_metric: float = caption_splitting_metrics(results=results)

    metrics: Dict[str, float] = {
        'levenstein_metric': lavenstein_metric
    }

    pprint(metrics)

    return metrics
