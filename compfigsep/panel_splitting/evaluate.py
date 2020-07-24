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


####################################################
Module to evaluate the panel splitting task metrics.
"""

from typing import Dict, Tuple, Iterable

from sortedcontainers import SortedKeyList
import numpy as np

from ..data.figure_generators import FigureGenerator
from ..utils.figure import Figure
from ..utils.average_precision import compute_average_precision


def panel_splitting_figure_eval(figure: Figure, stat_dict: Dict[str, any]):
    """
    Evaluate panel splitting metrics on a single figure.

    Args:
        figure (Figure):            The figure on which to evaluate the panel splitting task.
        stat_dict (Dict[str, any]): A dict containing panel splitting evaluation stats
                                        It will be updated by this function.
    """
    stat_dict['num_samples'] += 1

    # Perform matching on this figure
    # This tests whether a detected panel is true positive or false positive
    figure.match_detected_and_gt_panels_splitting_task()

    stat_dict['overall_gt_count'] += len(figure.gt_subfigures)
    stat_dict['overall_detected_count'] += len(figure.detected_panels)

    # TODO remove
    print("Number of GT panels :", len(figure.gt_subfigures))
    print("Number of detected panels :", len(figure.detected_panels))

    num_correct_imageclef = 0
    num_correct_iou_thresh = 0

    for detected_panel in figure.detected_panels:
        num_correct_imageclef += int(detected_panel.is_true_positive_overlap)

        num_correct_iou_thresh += int(detected_panel.is_true_positive_iou)

        # Add this detection in the sorted list
        stat_dict['detections'].add((detected_panel.detection_score,
                                     detected_panel.is_true_positive_iou))

    # TODO remove
    print("Number of imageCLEF correct panels :", num_correct_imageclef)
    print("Number of IoU correct panels :", num_correct_iou_thresh)

    # ImageCLEF accuracy (based on overlap 0.66 threshold)
    k = max(len(figure.gt_subfigures), len(figure.detected_panels))
    imageclef_accuracy = num_correct_imageclef / k

    stat_dict['sum_imageclef_accuracies'] += imageclef_accuracy


def panel_splitting_metrics(stat_dict: Dict[str, any]) -> Tuple[int, int, int]:
    """
    Evaluate the panel splitting metrics.

    Args:
        stat_dict (Dict[str, any]): A dict containing the stats gathered while looping over
                                        detections.

    Returns:
        imageclef_accuracy (int):   The ImageCLEF accuracy as presented in this paper:
                                        (http://ceur-ws.org/Vol-1179/CLEF2013wn-ImageCLEF-
                                        SecoDeHerreraEt2013b.pdf).
        precision (int):            Precision value (TP / TP + FP).
        recall (int):               Recall value (TP / TP + FP).
        mAP (int):                  Mean average precision value.
    """
    # 1) ImageCLEF accuracy
    imageclef_accuracy = stat_dict['sum_imageclef_accuracies'] / stat_dict['num_samples']

    # true_positives = [1, 0, 1, 1, 1, 0, 1, 0, 0...] with a lot of 1 hopefully ;)
    true_positives = [np.float(is_positive) for _, is_positive in stat_dict['detections']]
    overall_correct_count = np.sum(true_positives)

    # 2) overall recall = TP / TP + FN
    recall = overall_correct_count / stat_dict['overall_gt_count']
    # 3) overall precision = TP / TP + FP
    precision = overall_correct_count / stat_dict['overall_detected_count']

    # 4) mAP computation
    cumsum_true_positives = np.cumsum(true_positives)

    # cumulated_recalls
    cumulated_recalls = cumsum_true_positives / stat_dict['overall_gt_count']

    # = cumsum(TP + FP)
    cumsum_detections = np.arange(1, stat_dict['overall_detected_count'] + 1)
    cumulated_precisions = cumsum_true_positives / cumsum_detections

    # mAP = area under the precison/recall curve (only one 'class' here)
    mAP = compute_average_precision(recall=cumulated_recalls,
                                    precision=cumulated_precisions)

    return imageclef_accuracy, precision, recall, mAP


def evaluate_detections(figure_generator: Iterable[Figure]) -> Dict[str, float]:
    """
    Compute the metrics (ImageCLEF and mAP) from a given set of panel slitting detections.

    Args:
        figure_generator (Iterable[Figure]):    A figure generator yielding Figure objects
                                                    augmented with detected panels.

    Returns:
        metrics (Dict[str, float]): A dict containing the computed metrics.
    """
    stats = {
        'num_samples': 0,
        'overall_gt_count': 0,
        'overall_detected_count': 0,
        'detections': SortedKeyList(key=lambda u: -u[0]),
        'overall_correct_count': 0,
        'sum_imageclef_accuracies': 0
    }

    for figure in figure_generator:
        panel_splitting_figure_eval(figure, stats)

    imageclef_accuracy, precision, recall, mean_average_precision = panel_splitting_metrics(
        stat_dict=stats)

    print(f"ImageCLEF Accuracy (overlap threshold = 0.66): {imageclef_accuracy:.3f}\n"\
          f"Precision: {precision:.3f}\n"\
          f"Recall: {recall:.3f}\n"\
          f"mAP (IoU threshold = 0.5): {mean_average_precision:.3f}")

    metrics = {
        'image_clef_accuracy': imageclef_accuracy,
        'precision': precision,
        'recall': recall,
        'mAP': mean_average_precision
    }

    return metrics
