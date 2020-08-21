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

from typing import Tuple, Dict, List, Any
from pprint import pprint

from sortedcontainers import SortedKeyList # type: ignore
import numpy as np # type: ignore

from ..data.figure_generators import FigureGenerator
from ..utils.figure import Figure
from ..utils.figure.label import Label
from ..utils.average_precision import compute_average_precision


def label_recognition_figure_eval(figure: Figure,
                                  stat_dict: Dict[str, Any]) -> None:
    """
    Evaluate label recognition metrics on a single figure.

    Args:
        figure (Figure):            The figure on which to evaluate the panel splitting task.
        stat_dict (Dict[str, any]): A dict containing label recognition evaluation stats
                                        It will be updated by this function.
    """
    # Perform matching on this figure
    # This tests whether a detected label is true positive or false positive
    figure.match_detected_and_gt_labels()

    # TODO remove
    num_gt_labels: int = 0

    # Keep track of the number of gt labels for each class
    for gt_subfigure in figure.gt_subfigures:

        if gt_subfigure.label is None:
            continue

        gt_label: Label = gt_subfigure.label

        # Drop useless panels for this task
        if gt_label.box is None or gt_label.text is None or len(gt_label.text) != 1:
            continue

        # TODO remove
        num_gt_labels += 1
        cls = gt_label.text

        stat_dict['overall_gt_count'] += 1

        if cls not in stat_dict['gt_count_by_class']:
            stat_dict['gt_count_by_class'][cls] = 1
        else:
            stat_dict['gt_count_by_class'][cls] += 1


    stat_dict['overall_detected_count'] += len(figure.detected_labels)

    # TODO remove
    print("Number of GT labels :", num_gt_labels)
    print("Number of detected labels :", len(figure.detected_labels))

    num_correct = 0
    for detected_label in figure.detected_labels:
        num_correct += int(detected_label.is_true_positive)

        cls = detected_label.text

        # initialize the dict entry for this class if necessary
        # it is sorting the predictions in the decreasing order of their score
        if cls not in stat_dict['detections_by_class']:
            stat_dict['detections_by_class'][cls] = SortedKeyList(key=lambda u: -u[0])

        # Add this detection in the sorted list
        stat_dict['detections_by_class'][cls].add((detected_label.detection_score,
                                                   detected_label.is_true_positive))

    print("Number of correct detections :", num_correct)
    stat_dict['overall_correct_count'] += num_correct


def multi_class_metrics(stat_dict: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Compute the metrics for a multi class detection task.
    Used for both label recognition and panel segmentation tasks

    Args:
        stat_dict (Dict[str, any]): A dict containing the stats gathered while looping over
                                        detections.

    Returns:
        precision (float):  Precision value (TP / TP + FN).
        recall (float):     Recall value (TP / TP + FN).
        mAP (float):        Mean average precision value.
    """

    # 1) overall recall = TP / TP + FN
    recall: float = stat_dict['overall_correct_count'] / stat_dict['overall_gt_count']
    # 2) overall precision = TP / TP + FP
    precision: float = stat_dict['overall_correct_count'] / stat_dict['overall_detected_count']

    # 3) mean average precision (mAP)
    # Computation of mAP is done class wise
    mean_average_precision: float = 0
    for cls in stat_dict['detections_by_class']:

        # true_positives = [1, 0, 1, 1, 1, 0, 1, 0, 0...] with a lot of 1 hopefully ;)
        class_true_positives: List[np.float] = [np.float(is_positive)
                                                for _, is_positive
                                                in stat_dict['detections_by_class'][cls]]

        class_detected_count: int = len(stat_dict['detections_by_class'][cls])
        class_gt_count: int = stat_dict['gt_count_by_class'][cls] \
            if cls in stat_dict['gt_count_by_class'] else 0

        class_cumsum_true_positives: np.array = np.cumsum(class_true_positives)

        # cumulated_recalls
        if class_gt_count == 0:
            class_cumulated_recalls: np.array = np.zeros(shape=(class_detected_count,))
        else:
            class_cumulated_recalls = class_cumsum_true_positives / class_gt_count

        # = cumsum(TP + FP)
        class_cumsum_detections: np.array = np.arange(1, class_detected_count + 1)
        class_cumulated_precisions: np.array = \
            class_cumsum_true_positives / class_cumsum_detections

        # Add the AP score to the overall mAP total
        mean_average_precision += compute_average_precision(
            recall=class_cumulated_recalls,
            precision=class_cumulated_precisions)


    # normalize mAP by the number of classes
    mean_average_precision /= len(stat_dict['detections_by_class'])

    return precision, recall, mean_average_precision


def evaluate_detections(figure_generator: FigureGenerator) -> Dict[str, float]:
    """
    Compute the metrics (precision, recall and mAP) from a given set of label recognition
    detections.

    Args:
        figure_generator (FigureGenerator): A figure generator yielding Figure objects augmented
                                                with detected labels.

    Returns:
        metrics (Dict[str, float]): A dict containing the computed metrics.
    """

    stats = {
        'overall_gt_count': 0,
        'overall_detected_count': 0,
        'overall_correct_count': 0,
        # detections_by_class is like: {class -> [(score, is_tp)]}
        'detections_by_class': {},
        # gt_count_by_class {class -> number_of_gt}
        'gt_count_by_class': {}
    }

    for figure in figure_generator():

        label_recognition_figure_eval(figure, stats)


    precision, recall, mean_average_precision = multi_class_metrics(stat_dict=stats)

    metrics = {
        'precision': precision,
        'recall': recall,
        'mAP': mean_average_precision
    }

    pprint(metrics)

    return metrics
