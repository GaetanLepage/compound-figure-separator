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


###############################################
Evaluation tool for compound figure separation.

TODO
"""

from typing import Dict, Any, List
from pprint import pprint

from sortedcontainers import SortedKeyList # type: ignore

from ..data.figure_generators import FigureGenerator
from ..utils.figure import Figure
from ..utils.figure import Label

from ..panel_splitting.evaluate import(PanelSplittingFigureResult,
                                       panel_splitting_figure_eval,
                                       panel_splitting_metrics)

from ..label_recognition.evaluate import(MultiClassFigureResult,
                                         label_recognition_figure_eval,
                                         multi_class_metrics)

from ..panel_segmentation.evaluate import panel_segmentation_figure_eval

from ..caption_splitting.evaluate import (CaptionSplittingFigureResult,
                                          caption_splitting_figure_eval,
                                          caption_splitting_metrics)


def compound_figure_separation_figure_eval(figure: Figure,
                                           stat_dict: Dict[str, Any]):
    """
    Evaluate compound figure separation metrics on a single figure.

    Args:
        figure (Figure):            The figure on which to evaluate the compound figure separation
                                        task.
        stat_dict (Dict[str, any]): A dict containing compound figure evaluation evaluation stats
                                        It will be updated by this function.
"""
    # Keep track of the number of gt panels for each class
    for gt_subfigure in figure.gt_subfigures:

        # When panels don't have (valid) label annotation, just set None for label
        if gt_subfigure.label is None:
            gt_subfigure.label = Label(box=None,
                                       text='')
        gt_label: Label = gt_subfigure.label

        if gt_label.text is None or len(gt_label.text) != 1:
            # TODO: this choice might not be smart
            # '' stands for "no label"
            gt_label.text = ''

        cls: str = gt_label.text

        stat_dict['overall_gt_count'] += 1

        if cls not in stat_dict['gt_count_by_class']:
            stat_dict['gt_count_by_class'][cls] = 1
        else:
            stat_dict['gt_count_by_class'][cls] += 1


    stat_dict['overall_detected_count'] += len(figure.detected_subfigures)

    for detected_subfigure in figure.detected_subfigures:

        stat_dict['overall_correct_count'] += int(detected_subfigure.is_true_positive)

        cls = detected_subfigure.label.text

        # Initialize the dict entry for this class if necessary.
        # It is sorting the predictions in the decreasing order of their score.
        if cls not in stat_dict['detections_by_class']:
            stat_dict['detections_by_class'][cls] = SortedKeyList(key=lambda u: -u[0])

        # The subfigure detection score is set to be the same as the panel detection sore.
        detected_subfigure.detection_score = detected_subfigure.panel.detection_score

        # Add this detection in the sorted list.
        stat_dict['detections_by_class'][cls].add((detected_subfigure.detection_score,
                                                   detected_subfigure.is_true_positive))


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

    panel_splitting_results: List[PanelSplittingFigureResult] = []
    label_recognition_results: List[MultiClassFigureResult] = []
    panel_segmentation_results: List[MultiClassFigureResult] = []
    caption_splitting_results: List[CaptionSplittingFigureResult] = []

    for figure in figure_generator():

        # 1) Panel splitting
        panel_splitting_results.append(panel_splitting_figure_eval(figure))
        # print("\nPanel splitting figure stats")
        # figure.show_preview(mode='both', window_name='panel_splitting')

        # 2) Label recognition
        label_recognition_results.append(label_recognition_figure_eval(figure))
        # print("\nLabel recognition figure stats")
        # figure.show_preview(mode='both', window_name='label_recognition')

        # 3) Panel segmentation
        # Assign detected labels to detected panels using the beam search algorithm
        # TODO manage the case where no labels have been detected
        # if len(detected_labels) > 0:
        # subfigures = beam_search.assign_labels_to_panels(figure.detected_panels,
                                                         # figure.detected_labels)

        # Convert output from beam search (list of SubFigure objects) to DetectedSubFigure.
        # figure.detected_subfigures = [DetectedSubFigure.from_normal_sub_figure(subfigure)
                                      # for subfigure in subfigures]

        # print("\nPanel segmentation figure stats")
        panel_segmentation_results.append(panel_segmentation_figure_eval(figure))
        #figure.show_preview(mode='both', window_name='panel_segmentation')

        # 4) Caption segmentation
        caption_splitting_results.append(caption_splitting_figure_eval(figure))


    metrics: Dict[str, Dict[str, float]] = {}

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
    lrec_precision, lrec_recall, lrec_map = multi_class_metrics(
        results=label_recognition_results)
    metrics['label_recognition'] = {
        'precision': lrec_precision,
        'recall': lrec_recall,
        'mAP': lrec_map
    }

    # Panel segmentation
    pseg_precision, pseg_recall, pseg_map = multi_class_metrics(
        results=panel_segmentation_results)
    metrics['panel_segmentation'] = {
        'precision': pseg_precision,
        'recall': pseg_recall,
        'mAP': pseg_map
    }

    # Caption splitting
    lavenstein_metric: float = caption_splitting_metrics(caption_splitting_results)

    metrics['caption_splitting'] = {
        'lavenstein_metric': lavenstein_metric
    }

    pprint(metrics)

    return metrics['panel_segmentation']
