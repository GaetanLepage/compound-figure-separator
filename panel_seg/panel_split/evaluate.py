"""
Module to evaluate the panel splitting task metrics.
"""

from typing import List


def evaluate_predictions(figure_generator: str):
    """
    Compute the metrics (ImageCLEF and mAP) from a given set of panel slitting predictions.

    Args:
        figure_generator:   A figure generator yielding Figure objects augmented with
                                predicted panels.

    Returns:
        A dict containing the computed metrics.
    """

    num_samples = 0
    overall_gt_count = 0
    overall_pred_count = 0

    # ImageCLEF
    sum_imageclef_accuracies = 0.0

    # Stats to compute mAP
    overall_correct_count = 0
    sum_recalls = 0.0
    sum_precisions = 0.0


    for figure in figure_generator:

        # Common counters
        num_samples += 1

        overall_gt_count += len(figure.gt_panels)
        overall_pred_count += len(figure.pred_panels)

        # 1) ImageCLEF accuracy
        num_correct_imageclef = figure.get_num_correct_predictions(
            use_overlap_instead_of_iou=True)
        k = max(len(figure.gt_panels), len(figure.pred_panels))
        imageclef_accuracy = num_correct_imageclef / k

        sum_imageclef_accuracies += imageclef_accuracy

        # 2) Usual metrics
        num_correct_iou_thresh = figure.get_num_correct_predictions(
            use_overlap_instead_of_iou=False)
        overall_correct_count += num_correct_iou_thresh

        # recall = TP / TP + FN
        recall = num_correct_iou_thresh / len(figure.gt_panels)
        sum_recalls += recall

        # precision = TP / TP + FP
        if len(figure.pred_panels) == 0:
            precision = 0
        else:
            precision = num_correct_iou_thresh / len(figure.pred_panels)
        sum_precisions += precision


    imageclef_accuracy = sum_imageclef_accuracies / num_samples

    recall = overall_correct_count / overall_gt_count
    precision = overall_correct_count / overall_pred_count
    # mAP = TP / TP + FP
    mAP = overall_correct_count / overall_pred_count

    print(f"ImageCLEF Accuracy (overlap threshold = 0.66): {imageclef_accuracy:.3f}\n"\
            f"Precision: {precision:.3f}\n"\
            f"mAP (IoU threshold = 0.5): {mAP:.3f}")

    metrics = {
        'image_clef_accuracy': imageclef_accuracy,
        'precision': precision,
        'recall': recall,
        'mAP': mAP
        }

    return metrics
