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


#######################################
TODO remove this file
"""

from typing import List, Tuple
import numpy as np

from ..data.figure_generators import FigureGenerator
from ..utils.figure import Figure, Panel

Box = List[int, int, int, int]


def _overlap_with_mask(box: Box, mask: Box) -> float:
    """
    Compute the overlap of the box with the mask.

    Args:
        box (Box):  The coordinates of the bounding box.
        mask (Box): The mask.

    Returns:
        ratio (float):  The overlap of the box with the mask.
    """
    roi = mask[box[1]:box[3], box[0]:box[2]]
    count = np.count_nonzero(roi)
    box_area = (box[2] - box[0]) * (box[3]-box[1])
    ratio = count / box_area

    return ratio


def _update_mask(box: Box, mask: Box):
    """
    Changes the mask by setting all pixels of the `box` to 1.

    Args:
        box (Box):  A bounding box.
        mask (Box): The mask.
    """
    mask[box[1]:box[3], box[0]:box[2]] = 1


def _post_processing(boxes: List[Box],
                     scores: List[float],
                     labels: List[str],
                     image_shape: Tuple[int, int, int]):
    """
    Post processing to remove false positives.

    Args:
        boxes (List[Box]):                  A list of predicted bounding boxes.
        scores (List[float]):               A list of scores (one for each bbox).
        labels (List[str]):                 A list of labels.
        image_shape (Tuple[int, int, int]): The image shape (H, W, C).

    Returns:
        boxes (List[Box]):      The list of filtered bboxes.
        scores (List[float]):   The list of associated scores.
        labels (List[str]):     The list of associated labels.
    """
    # 1. We keep only scores greater than 0.05.
    indices = []
    for index, score in enumerate(scores):
        if score > 0.05:
            indices.append(index)
        # scores are sorted so we can break.
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


def predict(figure_generator: FigureGenerator,
            predict_function: callable,
            pre_processing_function: callable = None) -> Figure:
    """
    Predicts the sub figures locations (bounding boxes) and yields the augmented Figure object.

    Args:
        figure_generator (FigureGenerator): A generator yielding figures.
        predict_function (callable):        A function taking an image as input and predicting
                                                the panels locations and returning
                                                [boxes, scores, labels].
        pre_processing_function (callable): A function taking an image as input and applying
                                                preprocessing on this image.

    Yields:
        annotated Figure objects augmented with predicted panels.
    """

    for figure in figure_generator():

        # Load image.
        image = figure.image

        # Pre-processing.
        if pre_processing_function is not None:
            image = pre_processing_function(image)

        # Predicting.
        boxes, scores, labels = predict_function(image)

        # Post-processing.
        boxes, scores, labels = _post_processing(boxes=boxes,
                                                 scores=scores,
                                                 labels=labels,
                                                 image_shape=image.shape)

        # Export detections
        detected_panels = []
        for box in boxes:

            panel = SubFigure(panel_rect=box)

            detected_panels.append(panel)

        figure.detected_panels = detected_panels

        yield figure
