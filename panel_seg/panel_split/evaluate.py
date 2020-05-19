"""
Module to evaluate the panel splitting task metrics.
"""

from typing import List

from sortedcontainers import SortedKeyList
import numpy as np

from panel_seg.utils.average_precision import compute_average_precision


def evaluate_detections(figure_generator: str):
    """
    Compute the metrics (ImageCLEF and mAP) from a given set of panel slitting detections.

    Args:
        figure_generator:   A figure generator yielding Figure objects augmented with
                                detected panels.

    Returns:
        A dict containing the computed metrics.
    """

    num_samples = 0
    overall_gt_count = 0
    overall_detected_count = 0

    # ImageCLEF
    sum_imageclef_accuracies = 0.0

    # Stats to compute mAP
    overall_correct_count = 0

    # TODO explain this choice
    detections = SortedKeyList(key=lambda u: u[0])

    for figure in figure_generator:

        # Perform matching on this figure
        # This tests whether a detected panel is true positive or false positive
        figure.match_detected_and_gt_panels()

        # Common counters
        num_samples += 1
        overall_gt_count += len(figure.gt_panels)
        overall_detected_count += len(figure.detected_panels)

        num_correct_imageclef = 0
        num_correct_iou_thresh = 0

        for detected_panel in figure.detected_panels:
            num_correct_imageclef += int(detected_panel.panel_is_true_positive_overlap)

            num_correct_iou_thresh += int(detected_panel.panel_is_true_positive_iou)

            detections.add((detected_panel.panel_detection_score,
                            detected_panel.panel_is_true_positive_iou))


        # ImageCLEF accuracy (based on overlap 0.66 threshold)
        k = max(len(figure.gt_panels), len(figure.detected_panels))
        imageclef_accuracy = num_correct_imageclef / k

        sum_imageclef_accuracies += imageclef_accuracy

    # 1) ImageCLEF accuracy
    imageclef_accuracy = sum_imageclef_accuracies / num_samples

    # true_positives = [1, 0, 1, 1, 1, 0, 1, 0, 0...] with a lot of 1 hopefully ;)
    true_positives = [np.float(is_positive) for _, is_positive in detections]
    overall_correct_count = np.sum(true_positives)

    # 2) overall recall = TP / TP + FN
    recall = overall_correct_count / overall_gt_count
    # 3) overall precision = TP / TP + FP
    precision = overall_correct_count / overall_detected_count

    # 4) mAP computation
    cumsum_true_positives = np.cumsum(true_positives)

    # cumulated_recalls
    cumulated_recalls = cumsum_true_positives / overall_gt_count

    # = cumsum(TP + FP)
    cumsum_detections = np.arange(1, overall_detected_count + 1)
    cumulated_precisions = cumsum_true_positives / cumsum_detections

    # mAP = area under the precison/recall curve (only one 'class' here)
    mAP = compute_average_precision(rec=cumulated_recalls,
                                          prec=cumulated_precisions)


    print(f"ImageCLEF Accuracy (overlap threshold = 0.66): {imageclef_accuracy:.3f}\n"\
            f"Precision: {precision:.3f}\n"\
            f"Recall: {recall:.3f}\n"\
            f"mAP (IoU threshold = 0.5): {mAP:.3f}")

    metrics = {
        'image_clef_accuracy': imageclef_accuracy,
        'precision': precision,
        'recall': recall,
        'mAP': mAP
        }

    return metrics
