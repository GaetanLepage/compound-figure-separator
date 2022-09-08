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

from typing import Optional
from pprint import pprint
from collections import namedtuple

from joblib import Parallel, delayed

import textdistance

from ..utils.figure import Figure
from ..data.figure_generators import FigureGenerator

CaptionSplittingFigureResult: namedtuple = namedtuple(
    "CaptionSplittingFigureResult",
    [
        'normalized_levenshtein_similarity'
    ]
)


def caption_splitting_figure_eval(figure: Figure) -> Optional[CaptionSplittingFigureResult]:
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
    # Number of ground truth labels. i.e. the normalizer for the figure score.
    num_gt_labels: int = 0

    # Initialize the metrics values
    normalized_levenshtein_similarity: float = 0

    # Get the dictionnary of detected subcaptions (if it exists).
    detected_subcaptions: dict[str, str] = figure.detected_subcaptions \
        if hasattr(figure, 'detected_subcaptions') \
        else {}

    # If the figure was not annotated, return None.
    if not hasattr(figure, 'gt_subfigures'):
        return None

    # Case where this is not a compound figure.
    if len(figure.gt_subfigures) == 1 and '_' in detected_subcaptions:
        num_gt_labels = 1
        normalized_levenshtein_similarity = textdistance.levenshtein.normalized_similarity(
            figure.gt_subfigures[0],
            detected_subcaptions['_']
        )

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
                normalized_levenshtein_similarity += textdistance.levenshtein.normalized_similarity(
                    gt_subcaption,
                    detected_subcaption
                )

    if num_gt_labels > 0:
        # Normalize the socre over the number of GT subcaptions.
        normalized_levenshtein_similarity /= num_gt_labels

        return CaptionSplittingFigureResult(
            normalized_levenshtein_similarity=normalized_levenshtein_similarity
        )

    # If the ground truth figure was not annotated (no GT labels), return None.
    return None


def caption_splitting_metrics(results: list[CaptionSplittingFigureResult]) -> float:
    """
    Compute the metrics for the caption splitting task.

    Args:
        results (list[CaptionSplittingFigureResult]):   TODO.

    Returns:
        levenshtein_metric (float):  The averaged levenshtein metric.
    """
    num_captions: int = 0

    normalized_levenshtein_similarity: float = 0

    for result in results:
        # Filter out the figures for which no score was computed (missing annotations for e.g.).
        if result is None:
            continue

        num_captions += 1
        normalized_levenshtein_similarity += result.normalized_levenshtein_similarity

    # Normalize the score by the number of captions
    normalized_levenshtein_similarity /= num_captions

    return normalized_levenshtein_similarity


def evaluate_detections(figure_generator: FigureGenerator) -> dict[str, float]:
    """
    Compute the metrics from a given set of predicted sub captions.

    Args:
        figure_generator (FigureGenerator): A figure generator yielding Figure objects
                                                augmented with detected sub captions.

    Returns:
        metrics (dict[str, any]): A dict containing the computed metrics.
    """

    # Parallel computing to speed up the evaluation
    with Parallel(n_jobs=-1) as parallel:

        results: list[CaptionSplittingFigureResult] = parallel(
            [
                delayed(caption_splitting_figure_eval)(figure)
                for figure in figure_generator()
            ]
        )

    lavenstein_metric: float = caption_splitting_metrics(results=results)

    metrics: dict[str, float] = {
        'levenstein_metric': lavenstein_metric
    }

    pprint(metrics)

    return metrics
