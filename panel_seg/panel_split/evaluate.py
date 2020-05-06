"""
Module to evaluate the panel splitting task metrics.
"""


def evaluate_predictions(figure_generator: str):
    """
    Compute the metrics (ImageCLEF and mAP) from a given set of panel slitting predictions.

    Args:
        figure_generator:   A figure generator yielding Figure objects augmented with
                                predicted panels.
    """

    num_samples = 0

    overall_correct_count = 0
    overall_gt_count = 0
    overall_pred_count = 0

    sum_accuracies = 0.0
    sum_recalls = 0.0
    sum_precisions = 0.0


    for figure in figure_generator:

        # Compute the number of good predictions
        num_correct = figure.map_gt_and_predictions()


        k = max(len(figure.gt_panels), len(figure.pred_panels))
        accuracy = num_correct / k
        recall = num_correct / len(figure.gt_panels)
        if len(figure.pred_panels) == 0:
            precision = 0
        else:
            precision = num_correct / len(figure.pred_panels)

        sum_accuracies += accuracy
        sum_recalls += recall
        sum_precisions += precision

        overall_correct_count += num_correct
        overall_gt_count += len(figure.gt_panels)
        overall_pred_count += len(figure.pred_panels)

        num_samples += 1

    # ImageCLEF scores
    image_clef_accuracy = sum_accuracies / num_samples
    image_clef_precision = sum_precisions / num_samples
    image_clef_recall = sum_recalls / num_samples

    print("ImageCLEF Accuracy: {}, Precision: {}, Recall: {}\n".format(image_clef_accuracy,
                                                                       image_clef_precision,
                                                                       image_clef_recall))

    # Regular scores
    overall_accuracy = overall_correct_count / max(overall_pred_count, overall_gt_count)
    overall_recall = overall_correct_count / overall_gt_count
    overall_precision = overall_correct_count / overall_pred_count

    print("Overall Accuracy: {}, Precision: {}, Recall: {}\n".format(overall_accuracy,
                                                                     overall_precision,
                                                                     overall_recall))
