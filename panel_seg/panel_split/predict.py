"""
TODO
"""


from typing import List
import numpy as np

from panel_seg.utils.figure.panel import Panel


def _overlap_with_mask(box: List[float], mask: List[float]):
    """
    Compute the overlap of the box with the mask.

    Args:
        box:    The coordinates of the bounding box.
        mask:   The mask.
    """
    roi = mask[box[1]:box[3], box[0]:box[2]]
    count = np.count_nonzero(roi)
    box_area = (box[2] - box[0]) * (box[3]-box[1])
    ratio = count/box_area
    return ratio


def _update_mask(box, mask):
    """
    Changes the mask by setting all pixels of the `box` to 1.

    Args:
        box:    A bounding box.
        mask:   The mask.
    """
    mask[box[1]:box[3], box[0]:box[2]] = 1


def _post_processing(boxes: List[List[float]],
                     scores: List[float],
                     labels: List[str],
                     image_shape: List[float]):
    """
    Post processing to remove false positives.

    Args:
        boxes:          A list of predicted bounding boxes.
        scores:         A list of scores (one for each bbox).
        labels:         A list of labels.
        image_shape:    The image shape (H, W, C).

    Returns:
        boxes:      The list of filtered bboxes.
        scores:     The list of associated scores.
        labels:     The list of associated labels.
    """
    # 1. We keep only scores greater than 0.05
    indices = []
    for index, score in enumerate(scores):
        if score > 0.05:
            indices.append(index)
        # scores are sorted so we can break
        else:
            break

    boxes, scores, labels = boxes[indices], scores[indices], labels[indices]

    # 2. We remove boxes which overlaps with other high score boxes
    indices = []
    height, width, _ = image_shape
    mask = np.zeros((height, width))

    for index, box in enumerate(boxes):
        box_i = box.astype(int)
        if _overlap_with_mask(box_i, mask) < 0.66:
            indices.append(index)
            _update_mask(box_i, mask)
    boxes, scores, labels = boxes[indices], scores[indices], labels[indices]

    # 3. We keep only at  most 30 panels
    if len(boxes) > 30:
        boxes, scores, labels = boxes[:30], scores[:30], labels[:30]

    return boxes, scores, labels


def predict(figure_generator,
            predict_function,
            pre_processing_function=None):
    """
    Predicts the sub figures locations (bounding boxes) and yields the augmented Figure object.

    Args:
        figure_generator:           A generator yielding figures.
        predict_function:           A function taking an image as input and predicting the panels
                                        locations and returning [boxes, scores, labels]
        pre_processing_function:    A function taking an image as input and applying preprocessing
                                        on this image

    Yields:
        annotated Figure objects augmented with predicted panels.
    """

    for figure in figure_generator:

        # Load image
        image = figure.image

        if pre_processing_function is not None:
            image = pre_processing_function(image)

        boxes, scores, labels = predict_function(image)

        boxes, scores, labels = _post_processing(boxes=boxes,
                                                 scores=scores,
                                                 labels=labels,
                                                 image_shape=image.shape)

        pred_panels = []
        for box in boxes:

            # TODO check coordinates order convention
            panel = Panel(panel_rect=box)

            pred_panels.append(panel)

        figure.pred_panels = pred_panels

        yield figure
