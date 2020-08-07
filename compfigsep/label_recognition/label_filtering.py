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

Collaborators:  Niccolò Marini (niccolo.marini@hevs.ch)
                Stefano Marchesin (stefano.marchesin@unipd.it)


#################################################################
Function to filter out unlikely panel labels.

Based on the Belief-Propagation algorithm proposed in this paper:
https://lhncbc.nlm.nih.gov/system/files/pub2011-082.pdf
"""

from typing import List, Dict, Set

from ..utils.figure.label import (DetectedLabel,
                                  LabelStructureEnum,
                                  LABEL_INDEX)
from ..utils import box

ALPHA = 1
BETA = 1
GAMMA = 1
DELTA = 1


def _unary_compatibility(label: DetectedLabel) -> float:
    """
    TODO
    """
    assert label.detection_score is not None

    ratio_prior_label: float = 0 / 1 # TODO

    return ALPHA * ratio_prior_label + BETA * label.detection_score



def _binary_compatibility(label_1: DetectedLabel,
                          label_2: DetectedLabel,
                          label_structure: LabelStructureEnum) -> float:
    """
    TODO

    Args:
        label_1 (DetectedLabel):                TODO.
        label_2 (DetectedLabel):                TODO.
        label_structure (LabelStructureEnum):   TODO.
    """

    assert label_1.box is not None and label_2.box is not None
    assert label_1.text is not None and label_2.text is not None

    ratio_areas: float = box.area(label_1.box) / box.area(label_2.box)

    label_1_index: int = LABEL_INDEX[label_structure](label_1.text)
    label_2_index: int = LABEL_INDEX[label_structure](label_2.text)

    label_1_pos: box.Point = box.get_center(label_1.box)
    label_2_pos: box.Point = box.get_center(label_2.box)

    delta_x: int = label_2_pos.x - label_1_pos.x
    delta_y: int = label_2_pos.y - label_1_pos.y


    text_labels_relationship: bool

    # We consider horizontal relationship.
    if abs(delta_x) > abs(delta_y):

        # label_2_pos.x > label_1_pos.x
        # i.e. label_1 is at the left of label_2
        if delta_x > 0:

            # True if label_1 has a lower index than label_2
            text_labels_relationship = label_1_index < label_2_index

        # label_2_pos.x < label_1_pos.x
        # i.e. label_1 is at the right of label_2
        else:
            # True if label_1 has a greater index than label_2
            text_labels_relationship = label_1_index > label_2_index

    # We consider vertical relationship.
    else:

        # label_2_pos.y > label_1_pos.y
        # i.e. label_1 is above label_2
        if delta_y > 0:

            # True if label_1 has a lower index than label_2
            text_labels_relationship = label_1_index < label_2_index

        # label_2_pos.y < label_1_pos.y
        # i.e. label_1 is below label_2
        else:
            # True if label_1 has a greater index than label_2
            text_labels_relationship = label_1_index > label_2_index

    return GAMMA * ratio_areas + DELTA * text_labels_relationship



def _message_update_rule(label_from: DetectedLabel,
                         label_to: DetectedLabel) -> float:
    """
    TODO
    """

    prob_message: float

    return prob_message


def _build_neighbors(label_list: List[DetectedLabel]) -> Dict[DetectedLabel,
                                                              Set[DetectedLabel]]:
    """
    TODO
    """

    neighbors_sets: Dict[DetectedLabel, Set[DetectedLabel]] = {
        label: set() for label in label_list
    }

    for label in label_list:

        assert label.box is not None

        label_x_left, label_y_top, label_x_right, label_y_bot = label.box

        for potential_neighbor in label_list:

            if potential_neighbor == label:
                continue

            assert potential_neighbor.box is not None

            potential_neighbor_center: box.Point = box.get_center(potential_neighbor.box)

            is_aligned_horizontally: bool = potential_neighbor_center.x >= label_x_left \
                and potential_neighbor_center.x <= label_x_right

            is_aligned_vertically: bool = potential_neighbor_center.y >= label_y_top \
                and potential_neighbor_center.y <= label_y_bot

            if is_aligned_horizontally or is_aligned_vertically:
                neighbors_sets[label].add(potential_neighbor)

    return neighbors_sets


def _belief(label: DetectedLabel,
            neighbors: Set[DetectedLabel]) -> float:

    return _unary_compatibility(label) * max(_message_update_rule(label_from=label_from,
                                                                  label_to=label)
                                             for label_from in neighbors)


def _belief_propagation() -> None:
    """
    TODO
    """
    # TODO


def filter_labels(label_list: List[DetectedLabel]) -> None:
    """
    TODO

    Args:
        label_list (List[DetectedLabel]):   A list of detected labels.
    """

    raise NotImplementedError("TODO")
