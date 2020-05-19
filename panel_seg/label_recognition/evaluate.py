"""
Module to evaluate the panel splitting task metrics.
"""

from typing import List

from sortedcontainers import SortedKeyList
import numpy as np

from panel_seg.utils.average_precision import compute_average_precision

# TODO remove
from panel_seg.utils.box import iou


def evaluate_detections(figure_generator: str):
    """
    Compute the metrics (precision, recall and mAP) from a given set of
    label recognition detections.

    Args:
        figure_generator:   A figure generator yielding Figure objects augmented with
                                detected labels.

    Returns:
        A dict containing the computed metrics.
    """

    num_samples = 0
    overall_gt_count = 0
    overall_detected_count = 0

    # Stats to compute mAP
    overall_correct_count = 0

    # {class -> [(score, is_tp)]}
    detections_by_class = {}
    # {class -> number_of_gt}
    gt_count_by_class = {}

    for figure in figure_generator:

        # Perform matching on this figure
        # This tests whether a detected panel is true positive or false positive
        figure.match_detected_and_gt_labels()

        # Count number of figures in the whole dataset
        num_samples += 1

        # Keep track of the number of gt labels for each class
        for gt_panel in figure.gt_panels:

            # Drop useless panels for this task
            if gt_panel.label_rect is None or len(gt_panel.label) != 1:
                continue

            cls = gt_panel.label

            overall_gt_count += 1

            if cls not in gt_count_by_class:
                gt_count_by_class[cls] = 1
            else:
                gt_count_by_class[cls] += 1


        overall_detected_count += len(figure.detected_panels)
        for detected_panel in figure.detected_panels:
            overall_correct_count += int(detected_panel.label_is_true_positive)

            cls = detected_panel.label

            # initialize the dict entry for this class if necessary
            if cls not in detections_by_class:
                detections_by_class[cls] = SortedKeyList(key=lambda u: u[0])

            detections_by_class[cls].add((detected_panel.label_detection_score,
                                          detected_panel.label_is_true_positive))


    # 1) overall recall = TP / TP + FN
    recall = overall_correct_count / overall_gt_count
    # 2) overall precision = TP / TP + FP
    precision = overall_correct_count / overall_detected_count

    # Computation of mAP is done class wise
    mAP = 0
    for cls in detections_by_class:

        # true_positives = [1, 0, 1, 1, 1, 0, 1, 0, 0...] with a lot of 1 hopefully ;)
        class_true_positives = [np.float(is_positive)
                                for _, is_positive in detections_by_class[cls]]

        class_detected_count = len(detections_by_class[cls])
        class_gt_count = gt_count_by_class[cls] if cls in gt_count_by_class else 0

        # 4) mAP computation
        class_cumsum_true_positives = np.cumsum(class_true_positives)

        # cumulated_recalls
        if class_gt_count == 0:
            class_cumulated_recalls = np.zeros(shape=(class_detected_count,))
        else:
            class_cumulated_recalls = class_cumsum_true_positives / class_gt_count

        # = cumsum(TP + FP)
        class_cumsum_detections = np.arange(1, class_detected_count + 1)
        class_cumulated_precisions = class_cumsum_true_positives / class_cumsum_detections

        mAP += compute_average_precision(rec=class_cumulated_recalls,
                                         prec=class_cumulated_precisions)


    # normalize mAP by the number of classes
    mAP /= len(detections_by_class)

    print(f"Precision: {precision:.3f}\n"\
          f"Recall: {recall:.3f}\n"\
          f"mAP (IoU threshold = 0.5): {mAP:.3f}")

    metrics = {
        'precision': precision,
        'recall': recall,
        'mAP': mAP
        }

    return metrics
