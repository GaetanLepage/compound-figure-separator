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


####################################################
Module to evaluate the panel splitting task metrics.
"""
from collections import namedtuple

import numpy as np

from ..data.figure_generators import FigureGenerator
from ..utils.figure import Figure, DetectedPanel
from ..utils.average_precision import compute_average_precision
from . import panel_filtering


Detection = namedtuple(
    "Detection",
    [
        'score',
        'is_true_positive'
    ]
)
PanelSplittingFigureResult = namedtuple(
    "PanelSplittingFigureResult",
    [
        'gt_count',
        'detected_count',
        'imageclef_accuracy',
        'detections'
    ]
)


def panel_splitting_figure_eval(figure: Figure) -> PanelSplittingFigureResult:
    """
    Evaluate panel splitting metrics on a single figure.

    Args:
        figure (Figure):            The figure on which to evaluate the panel splitting task.

    Returns:
        result (PanelSplittingFigureResult):    TODO
    """
    # Perform matching on this figure
    # This tests whether a detected panel is true positive or false positive
    figure.match_detected_and_gt_panels_splitting_task()

    num_correct_imageclef: int = 0
    num_correct_iou_thresh: int = 0

    detections: list[Detection] = []

    for detected_panel in figure.detected_panels:
        if detected_panel.is_true_positive_overlap is None \
                or detected_panel.is_true_positive_iou is None:
            raise ValueError("It seems like the evaluation process was not done"
                             " (is_true_positive_* attributes not set)")

        num_correct_imageclef += int(detected_panel.is_true_positive_overlap)

        num_correct_iou_thresh += int(detected_panel.is_true_positive_iou)

        # Add this detection in the list
        detections.append(Detection(score=detected_panel.detection_score,
                                    is_true_positive=detected_panel.is_true_positive_iou))

    # ImageCLEF accuracy (based on overlap 0.66 threshold)
    k: int = max(len(figure.gt_subfigures), len(figure.detected_panels))
    imageclef_accuracy: float = num_correct_imageclef / k

    return PanelSplittingFigureResult(
        gt_count=len(figure.gt_subfigures),
        detected_count=len(figure.detected_panels),
        imageclef_accuracy=imageclef_accuracy,
        detections=detections
    )


def panel_splitting_metrics(results: list[PanelSplittingFigureResult]
                            ) -> tuple[float, float, float, float]:
    """
    Evaluate the panel splitting metrics.

    Args:
        results (list[PanelSplittingFigureResult]): TODO.

    Returns:
        imageclef_accuracy (int):   The ImageCLEF accuracy as presented in this paper:
                                        (http://ceur-ws.org/Vol-1179/CLEF2013wn-ImageCLEF-
                                        SecoDeHerreraEt2013b.pdf).
        precision (int):            Precision value (TP / TP + FP).
        recall (int):               Recall value (TP / TP + FP).
        mAP (int):                  Mean average precision value.
    """
    # 0) Initialize statistics variables.
    overall_gt_count: int = 0
    overall_detected_count: int = 0
    sum_imageclef_accuracies: float = 0.0
    detections: list[Detection] = []

    # Loop through the results to compute statistics.
    for figure_result in results:
        overall_gt_count += figure_result.gt_count
        overall_detected_count += figure_result.detected_count

        sum_imageclef_accuracies += figure_result.imageclef_accuracy

        detections.extend(figure_result.detections)

    # Sort detections by decreasing detection scores
    detections.sort(reverse=True,
                    key=lambda detection: detection.score)

    # 1) ImageCLEF accuracy
    imageclef_accuracy: float = sum_imageclef_accuracies / len(results)

    # true_positives = [1, 0, 1, 1, 1, 0, 1, 0, 0...] with a lot of 1 hopefully ;)
    true_positives: list[int] = [int(detection.is_true_positive)
                                 for detection in detections]

    overall_correct_count: int = sum(true_positives)

    # 2) overall recall = TP / TP + FN
    recall: float = overall_correct_count / overall_gt_count
    # 3) overall precision = TP / TP + FP
    precision: float = overall_correct_count / overall_detected_count

    # 4) mAP computation
    cumsum_true_positives: np.ndarray = np.cumsum(true_positives)

    # cumulated_recalls
    cumulated_recalls: np.ndarray = cumsum_true_positives / overall_gt_count

    # cumulated_precisions
    cumsum_detections: np.ndarray = np.arange(1, overall_detected_count + 1)
    cumulated_precisions: np.ndarray = cumsum_true_positives / cumsum_detections

    # mAP = area under the precison/recall curve (only one 'class' here)
    mean_average_precision: float = compute_average_precision(recall=cumulated_recalls,
                                                              precision=cumulated_precisions)

    return imageclef_accuracy, precision, recall, mean_average_precision


def evaluate_detections(figure_generator: FigureGenerator) -> dict[str, float]:
    """
    Compute the metrics (ImageCLEF and mAP) from a given set of panel slitting detections.

    Args:
        figure_generator (Iterable[Figure]):    A figure generator yielding Figure objects
                                                    augmented with detected panels.

    Returns:
        metrics (dict[str, float]): A dict containing the computed metrics.
    """
    for figure in figure_generator():

        filtered_panels: list[DetectedPanel] = panel_filtering.filter_panels(
            panel_list=figure.detected_panels
        )
        figure.detected_panels = filtered_panels
    # list containing the evaluation statistics for each figure.
    results: list[PanelSplittingFigureResult] = [
        panel_splitting_figure_eval(figure)
        for figure in figure_generator()
    ]

    imageclef_accuracy, precision, recall, mean_average_precision = panel_splitting_metrics(
        results=results
    )

    print(f"ImageCLEF Accuracy (overlap threshold = 0.66): {imageclef_accuracy:.3f}\n"
          f"Precision: {precision:.3f}\n"
          f"Recall: {recall:.3f}\n"
          f"mAP (IoU threshold = 0.5): {mean_average_precision:.3f}")

    metrics: dict[str, float] = {
        'image_clef_accuracy': imageclef_accuracy,
        'precision': precision,
        'recall': recall,
        'mAP': mean_average_precision
    }

    return metrics
