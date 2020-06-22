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


#######################################################
Module to evaluate the panel segmentation task metrics.
"""

from typing import Dict
from pprint import pprint
from sortedcontainers import SortedKeyList

from ..panel_splitting.evaluate import panel_splitting_figure_eval, panel_splitting_metrics
from ..label_recognition.evaluate import label_recognition_figure_eval, multi_class_metrics
from ..utils.figure.figure import Figure
from ..utils.figure import beam_search


def panel_segmentation_figure_eval(figure: Figure, stat_dict: Dict[str, any]):
    """
    Evaluate panel segmentation metrics on a single figure.

    Args:
        figure (Figure):            The figure on which to evaluate the panel splitting task.
        stat_dict (Dict[str, any]): A dict containing panel segmentation evaluation stats
                                        It will be updated by this function.
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
        'panel_splitting': {
            'num_samples': 0,
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

        # 1) Panel splitting
        figure.detected_panels = figure.raw_detected_panels
        print("\nPanel splitting figure stats")
        panel_splitting_figure_eval(figure, stats['panel_splitting'])
        detected_panels = figure.detected_panels
        # pprint(stats['panel_splitting'])
        # figure.show_preview(mode='both', window_name='panel_splitting')

        # 2) Label recognition
        figure.detected_panels = figure.raw_detected_labels
        # print("\nLabel recognition figure stats")
        label_recognition_figure_eval(figure, stats['label_recognition'])
        detected_labels = figure.detected_panels
        # pprint(stats['label_recognition'])
        # figure.show_preview(mode='both', window_name='label_recognition')

        # 3) Panel segmentation
        # Assign detected labels to detected panels using the beam search algorithm
        # TODO manage the case where no labels have been detected
        # if len(detected_labels) > 0:
        beam_search.assign_labels_to_panels(detected_panels,
                                            detected_labels)

        figure.detected_panels = detected_panels

        #print("\nPanel segmentation figure stats")
        panel_segmentation_figure_eval(figure, stats['panel_segmentation'])
        #pprint(stats['panel_segmentation'])
        #figure.show_preview(mode='both', window_name='panel_segmentation')


    metrics = {}

    # Panel splitting
    psp_imageclef_acc, psp_precision, psp_recall, psp_map = panel_splitting_metrics(
        stat_dict=stats['panel_splitting'])
    metrics['panel_splitting'] = {
        'imageclef_accuracy': psp_imageclef_acc,
        'precision': psp_precision,
        'recall': psp_recall,
        'mAP': psp_map
    }

    # Label recognition
    lrec_precision, lrec_recall, lrec_map = multi_class_metrics(stats['label_recognition'])
    metrics['label_recognition'] = {
        'precision': lrec_precision,
        'recall': lrec_recall,
        'mAP': lrec_map
    }

    # Panel segmentation
    pseg_precision, pseg_recall, pseg_map = multi_class_metrics(stats['panel_segmentation'])
    metrics['panel_segmentation'] = {
        'precision': pseg_precision,
        'recall': pseg_recall,
        'mAP': pseg_map
    }

    pprint(metrics)

    return metrics
