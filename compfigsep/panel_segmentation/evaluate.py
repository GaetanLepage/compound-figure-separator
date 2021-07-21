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


#######################################################
Module to evaluate the panel segmentation task metrics.
"""

import logging
from pprint import pprint

from ..data.figure_generators import FigureGenerator
from ..panel_splitting import panel_filtering
from ..panel_splitting.evaluate import panel_splitting_figure_eval, panel_splitting_metrics
from ..panel_splitting.evaluate import Detection, PanelSplittingFigureResult
from ..label_recognition import label_filtering
from ..label_recognition.evaluate import (MultiClassFigureResult,
                                          label_recognition_figure_eval,
                                          multi_class_metrics)
from ..utils.figure import Figure
from ..utils.figure import Label
from ..utils.figure import beam_search


def panel_segmentation_figure_eval(figure: Figure) -> MultiClassFigureResult:
    """
    Evaluate panel segmentation metrics on a single figure.

    Args:
        figure (Figure):            The figure on which to evaluate the panel segmentation task.

    Returns:
        result (MultiClassFigureResult):    TODO.
    """
    # Perform matching on this figure
    # This tests whether a detected panel is true positive or false positive
    figure.match_detected_and_gt_panels_segmentation_task()

    gt_count: int = 0
    gt_count_by_class: dict[str, int] = {}

    # Keep track of the number of gt panels for each class
    for gt_subfigure in figure.gt_subfigures:

        # When panels don't have (valid) label annotation, just set None for label
        if gt_subfigure.label is None:
            gt_subfigure.label = Label(box=None,
                                       text='')
        gt_label: Label = gt_subfigure.label

        if len(gt_label.text) != 1:
            # TODO: this choice might not be smart
            # '' stands for "no label"
            gt_label.text = ''

        cls: str = gt_label.text

        gt_count += 1

        if cls not in gt_count_by_class:
            gt_count_by_class[cls] = 1
        else:
            gt_count_by_class[cls] += 1

    detected_count: int = len(figure.detected_subfigures)

    num_correct: int = 0
    detections_by_class: dict[str, list[Detection]] = {}

    for detected_subfigure in figure.detected_subfigures:

        if detected_subfigure.panel is None or detected_subfigure.label is None:
            logging.warning("Detected subfigure does not have both panel and label attributes.")
            continue

        if detected_subfigure.is_true_positive is None:
            logging.warning("The `is_true_positive` attribute is None for this detected"
                            " subfigure.")
            continue

        num_correct += int(detected_subfigure.is_true_positive)

        if detected_subfigure.label.text is None:
            logging.warning("The label of this subfigure doesn't have any text.")
            continue

        cls = detected_subfigure.label.text

        # Initialize the dict entry for this class if necessary.
        # It is sorting the predictions in the decreasing order of their score.
        if cls not in detections_by_class:
            detections_by_class[cls] = []

        # The subfigure detection score is set to be the same as the panel detection sore.
        detected_subfigure.detection_score = detected_subfigure.panel.detection_score

        # Add this detection in the sorted list.
        detections_by_class[cls].append(
            Detection(score=detected_subfigure.detection_score,
                      is_true_positive=detected_subfigure.is_true_positive))

    return MultiClassFigureResult(gt_count=gt_count,
                                  gt_count_by_class=gt_count_by_class,
                                  detected_count=detected_count,
                                  detections_by_class=detections_by_class,
                                  correct_count=num_correct)


def evaluate_detections(figure_generator: FigureGenerator) -> dict:
    """
    Compute the metrics (precision, recall and mAP) from a given set of panel segmentation
    detections.

    Args:
        figure_generator (Iterable[Figure]):    A figure generator yielding Figure objects
                                                    augmented with detected panels and labels.

    Returns:
        metrics (dict): A dict containing the computed metrics.
    """

    panel_splitting_results: list[PanelSplittingFigureResult] = []
    label_recognition_results: list[MultiClassFigureResult] = []
    panel_segmentation_results: list[MultiClassFigureResult] = []

    for figure in figure_generator():

        # 1) Panel splitting
        figure.detected_panels = panel_filtering.filter_panels(panel_list=figure.detected_panels)
        panel_splitting_results.append(panel_splitting_figure_eval(figure))
        # print("\nPanel splitting figure stats")
        # pprint(stats['panel_splitting'])
        # figure.show_preview(mode='both', window_name='panel_splitting')

        # 2) Label recognition
        print("###############")
        print(figure.image_filename)
        figure.detected_labels = label_filtering.filter_labels(label_list=figure.detected_labels)
        label_recognition_results.append(label_recognition_figure_eval(figure))
        # print("\nLabel recognition figure stats")
        # pprint(stats['label_recognition'])
        # figure.show_preview(mode='both', window_name='label_recognition')

        # 3) Panel segmentation

        # Assign detected labels to detected panels using the beam search algorithm
        if not hasattr(figure, 'detected_subfigures') or not figure.detected_subfigures:
            figure.match_detected_visual_panels_and_labels()

        # figure.show_preview(mode='both')
        # panel_segmentation_results.append(panel_segmentation_figure_eval(figure))
        # TODO manage the case where no labels have been detected
        # Convert output from beam search (list of SubFigure objects) to DetectedSubFigure.
        # figure.detected_subfigures = [DetectedSubFigure.from_normal_sub_figure(subfigure)
        # for subfigure in subfigures]

        # print("\nPanel segmentation figure stats")
        # pprint(stats['panel_segmentation'])
        # figure.show_preview(mode='both', window_name='panel_segmentation')

    metrics: dict[str, dict[str, float]] = {}

    # Panel splitting
    psp_imageclef_acc, psp_precision, psp_recall, psp_map = panel_splitting_metrics(
        results=panel_splitting_results)
    metrics['panel_splitting'] = {
        'imageclef_accuracy': psp_imageclef_acc,
        'precision': psp_precision,
        'recall': psp_recall,
        'mAP': psp_map
    }

    # Label recognition
    lrec_precision, lrec_recall, lrec_map = multi_class_metrics(results=label_recognition_results)
    metrics['label_recognition'] = {
        'precision': lrec_precision,
        'recall': lrec_recall,
        'mAP': lrec_map
    }

    # Panel segmentation
    # pseg_precision, pseg_recall, pseg_map = multi_class_metrics(results=panel_segmentation_results)
    # metrics['panel_segmentation'] = {
        # 'precision': pseg_precision,
        # 'recall': pseg_recall,
        # 'mAP': pseg_map
    # }
    # TODO remove
    metrics['panel_segmentation'] = {}

    pprint(metrics)

    return metrics['panel_segmentation']
