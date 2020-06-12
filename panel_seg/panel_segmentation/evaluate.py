"""
Module to evaluate the panel splitting task metrics.
"""

from sortedcontainers import SortedKeyList
from typing import Tuple
import numpy as np

from panel_seg.utils.average_precision import compute_average_precision
from panel_seg.utils.figure.figure import Figure

import panel_seg.utils.figure.beam_search as beam_search

# TODO remove
from pprint import pprint


def panel_splitting_figure_eval(figure: Figure, stat_dict: dict):
    """
    Evaluate panel splitting metrics on a single figure.

    Args:
        figure (Figure):    The figure on which to evaluate the panel splitting task.
        stat_dict (dict):   A dict containing panel splitting evaluation stats
                                It will be updated by this function.
    """

    # TODO make sure panels have been set correctly
    # Perform matching on this figure
    # This tests whether a detected panel is true positive or false positive
    figure.match_detected_and_gt_panels_splitting_task()

    stat_dict['overall_gt_count'] += len(figure.gt_panels)
    stat_dict['overall_detected_count'] += len(figure.detected_panels)

    # TODO remove
    print("Number of GT panels :", len(figure.gt_panels))
    print("Number of detected panels :", len(figure.detected_panels))

    num_correct_imageclef = 0
    num_correct_iou_thresh = 0

    for detected_panel in figure.detected_panels:
        num_correct_imageclef += int(detected_panel.panel_is_true_positive_overlap)

        num_correct_iou_thresh += int(detected_panel.panel_is_true_positive_iou)

        # Add this detection in the sorted list
        stat_dict['detections'].add((detected_panel.panel_detection_score,
                                     detected_panel.panel_is_true_positive_iou))

    # TODO remove
    print("Number of imageCLEF correct panels :", num_correct_imageclef)
    print("Number of IoU correct panels :", num_correct_iou_thresh)

    # ImageCLEF accuracy (based on overlap 0.66 threshold)
    k = max(len(figure.gt_panels), len(figure.detected_panels))
    imageclef_accuracy = num_correct_imageclef / k

    stat_dict['sum_imageclef_accuracies'] += imageclef_accuracy


def panel_splitting_metrics(stat_dict: dict, num_samples: int) -> Tuple[int, int, int]:
    """
    Evaluate the panel splitting metrics.

    Args:
        stat_dict (dict):   A dict containing the stats gathered while looping over detections.
        num_samples (int):  The number of samples in the data set.

    Returns:
        imageclef_accuracy, precision, recall, mAP (tuple): The final panel splitting metrics
    """
    # 1) ImageCLEF accuracy
    imageclef_accuracy = stat_dict['sum_imageclef_accuracies'] / num_samples

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



def label_recognition_figure_eval(figure: Figure, stat_dict: dict):
    """
    Evaluate label recognition metrics on a single figure.

    Args:
        figure (Figure):    The figure on which to evaluate the panel splitting task.
        stat_dict (dict):   A dict containing panel splitting evaluation stats
                                It will be updated by this function.
    """
    # TODO make sure panels have been set correctly
    # Perform matching on this figure
    # This tests whether a detected label is true positive or false positive
    figure.match_detected_and_gt_labels()

    # Keep track of the number of gt labels for each class
    for gt_panel in figure.gt_panels:

        # Drop useless panels for this task
        if gt_panel.label_rect is None or len(gt_panel.label) != 1:
            continue

        cls = gt_panel.label

        stat_dict['overall_gt_count'] += 1

        if cls not in stat_dict['gt_count_by_class']:
            stat_dict['gt_count_by_class'][cls] = 1
        else:
            stat_dict['gt_count_by_class'][cls] += 1


    stat_dict['overall_detected_count'] += len(figure.detected_panels)

    # TODO remove
    print("Number of GT labels :", len(figure.gt_panels))
    print("Number of detected labels :", len(figure.detected_panels))

    num_correct = 0
    for detected_panel in figure.detected_panels:
        num_correct += int(detected_panel.label_is_true_positive)

        cls = detected_panel.label

        # initialize the dict entry for this class if necessary
        # it is sorting the predictions in the decreasing order of their score
        if cls not in stat_dict['detections_by_class']:
            stat_dict['detections_by_class'][cls] = SortedKeyList(key=lambda u: -u[0])

        # Add this detection in the sorted list
        stat_dict['detections_by_class'][cls].add((detected_panel.label_detection_score,
                                                   detected_panel.label_is_true_positive))

    print("Number of correct detections :", num_correct)
    stat_dict['overall_correct_count'] += num_correct


def multi_class_metrics(stat_dict: dict) -> Tuple[int, int, int]:
    """
    TODO
    Used for both label recognition and panel segmentation tasks
    """

    # 1) overall recall = TP / TP + FN
    recall = stat_dict['overall_correct_count'] / stat_dict['overall_gt_count']
    # 2) overall precision = TP / TP + FP
    precision = stat_dict['overall_correct_count'] / stat_dict['overall_detected_count']

    # 3) mean average precision (mAP)
    # Computation of mAP is done class wise
    mAP = 0
    for cls in stat_dict['detections_by_class']:

        # true_positives = [1, 0, 1, 1, 1, 0, 1, 0, 0...] with a lot of 1 hopefully ;)
        class_true_positives = [np.float(is_positive)
                                for _, is_positive in stat_dict['detections_by_class'][cls]]

        class_detected_count = len(stat_dict['detections_by_class'][cls])
        class_gt_count = stat_dict['gt_count_by_class'][cls] \
            if cls in stat_dict['gt_count_by_class'] else 0

        class_cumsum_true_positives = np.cumsum(class_true_positives)

        # cumulated_recalls
        if class_gt_count == 0:
            class_cumulated_recalls = np.zeros(shape=(class_detected_count,))
        else:
            class_cumulated_recalls = class_cumsum_true_positives / class_gt_count

        # = cumsum(TP + FP)
        class_cumsum_detections = np.arange(1, class_detected_count + 1)
        class_cumulated_precisions = class_cumsum_true_positives / class_cumsum_detections

        # Add the AP score to the overall mAP total
        mAP += compute_average_precision(recall=class_cumulated_recalls,
                                         precision=class_cumulated_precisions)


    # normalize mAP by the number of classes
    mAP /= len(stat_dict['detections_by_class'])

    return precision, recall, mAP


def panel_segmentation_figure_eval(figure: Figure, stat_dict: dict):
    """
    TODO
    """
    # Keep track of the number of gt panels for each class
    for gt_panel in figure.gt_panels:

        # When panels don't have (valid) label annotation, just set None for label
        if gt_panel.label_rect is None or len(gt_panel.label) != 1:
            # TODO: this choice might not be smart
            # `None` stands for "no label"
            gt_panel.label = None

        cls = gt_panel.label

        stat_dict['overall_gt_count'] += 1

        if cls not in stat_dict['gt_count_by_class']:
            stat_dict['gt_count_by_class'][cls] = 1
        else:
            stat_dict['gt_count_by_class'][cls] += 1


    stat_dict['overall_detected_count'] += len(figure.detected_panels)
    for detected_panel in figure.detected_panels:
        stat_dict['overall_correct_count'] += int(detected_panel.label_is_true_positive)

        cls = detected_panel.label

        # initialize the dict entry for this class if necessary
        # it is sorting the predictions in the decreasing order of their score
        if cls not in stat_dict['detections_by_class']:
            stat_dict['detections_by_class'][cls] = SortedKeyList(key=lambda u: -u[0])

        # Add this detection in the sorted list
        stat_dict['detections_by_class'][cls].add((detected_panel.panel_detection_score,
                                                   detected_panel.panel_is_true_positive_iou))


def evaluate_detections(figure_generator: iter) -> dict:
    """
    Compute the metrics (precision, recall and mAP) from a given set of panel segmentation
    detections.

    Args:
        figure_generator (iter):   A figure generator yielding Figure objects augmented with
                                        detected panels and labels.

    Returns:
        metrics (dict): A dict containing the computed metrics.
    """

    stats = {
        'num_samples': 0,
        'panel_splitting': {
            'overall_gt_count': 0,
            'overall_detected_count': 0,
            'detections': SortedKeyList(key=lambda u: -u[0]),
            'overall_correct_count': 0,
            'sum_imageclef_accuracies': 0},
        'label_recognition': {
            'overall_gt_count': 0,
            'overall_detected_count': 0,
            'overall_correct_count': 0,
            # detections_by_class is like: {class -> [(score, is_tp)]}
            'detections_by_class': {},
            # gt_count_by_class {class -> number_of_gt}
            'gt_count_by_class': {}},
        'panel_segmentation': {
            'overall_gt_count': 0,
            'overall_detected_count': 0,
            'overall_correct_count': 0,
            # detections_by_class is like: {class -> [(score, is_tp)]}
            'detections_by_class': {},
            # gt_count_by_class {class -> number_of_gt}
            'gt_count_by_class': {}}
    }


    for figure in figure_generator:

        # Count number of figures in the whole dataset
        stats['num_samples'] += 1

        # print("##############################")

        # 1) Panel splitting evaluation
        figure.detected_panels = figure.raw_detected_panels
        # print("\nPanel splitting figure stats")
        panel_splitting_figure_eval(figure, stats['panel_splitting'])
        # detected_panels = figure.detected_panels
        # pprint(stats['panel_splitting'])
        # figure.show_preview(mode='both', window_name='panel_splitting')

        # 2) Label recognition evaluation
        figure.detected_panels = figure.raw_detected_labels
        # print("\nLabel recognition figure stats")
        label_recognition_figure_eval(figure, stats['label_recognition'])
        # detected_labels = figure.detected_panels
        # pprint(stats['label_recognition'])
        # figure.show_preview(mode='both', window_name='label_recognition')

        # 3) Panel segmentation recognition

        # Assign detected labels to detected panels using the beam search algorithm
        # TODO manage the case where no labels have been detected
        #if len(detected_labels) > 0:
        #    beam_search.assign_labels_to_panels(detected_panels,
        #                                        detected_labels)

        #figure.detected_panels = detected_panels

        #print("\nPanel segmentation figure stats")
        #panel_segmentation_figure_eval(figure, stats['panel_segmentation'])
        #pprint(stats['panel_segmentation'])
        #figure.show_preview(mode='both', window_name='panel_segmentation')


    metrics = {}

    psp_imageclef_acc, psp_precision, psp_recall, psp_map = panel_splitting_metrics(
        stats['panel_splitting'])
    metrics['panel_splitting'] = {
        'imageclef_accuracy': psp_imageclef_acc,
        'precision': psp_precision,
        'recall': psp_recall,
        'mAP': psp_map
    }

    lrec_precision, lrec_recall, lrec_map = multi_class_metrics(stats['label_recognition'])
    metrics['label_recognition'] = {
        'precision': lrec_precision,
        'recall': lrec_recall,
        'mAP': lrec_map
    }

    # pseg_precision, pseg_recall, pseg_map = multi_class_metrics(stats['panel_segmentation'])
    # metrics['panel_segmentation'] = {
        # 'precision': pseg_precision,
        # 'recall': pseg_recall,
        # 'mAP': pseg_map
    # }


    return metrics
