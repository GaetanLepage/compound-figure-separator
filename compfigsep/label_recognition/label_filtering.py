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


#################################################################
Function to filter out unlikely panel labels.

Based on the Belief-Propagation algorithm proposed in this paper:
https://lhncbc.nlm.nih.gov/system/files/pub2011-082.pdf
"""

from typing import List, Dict, Set, Tuple
import numpy as np

from ..utils.figure.label import (DetectedLabel,
                                  LabelStructure,
                                  LabelStructureEnum,
                                  LABEL_INDEX)

from ..utils import box

ALPHA = 1
BETA = 1
GAMMA = 1
DELTA = 1


def _unary_compatibility(label: DetectedLabel,
                         label_index: int,
                         number_of_labels_in_prior_zone: int) -> np.ndarray:
    """
    Computes the unary compatibility function values for the detected label for both possible
    values of assigned label f_i.

    Args:
        label (DetectedLabel):                  A detected label.
        label_1_index (int):                    The index (within the LabelStructure) of the label.
                                                    Example, index of 'C' is 3.
        number_of_labels_in_prior_zone (int):   The number of labels in the 'prior label zone'
                                                    regarding label.

    Returns:
        result (np.ndarray):    The values of the unary compatibility function for both of the
                                    possible input values f_i.
                                    -> Shape = (2)
    """
    assert label.detection_score is not None

    result: np.ndarray = np.zeros(shape=(2))

    ratio_prior_label: float = number_of_labels_in_prior_zone / (label_index - 1)

    result[1] = ALPHA * ratio_prior_label + BETA * label.detection_score
    result[0] = 1 - result[1]

    return result



def _binary_compatibility(label_1: DetectedLabel,
                          label_2: DetectedLabel,
                          label_1_index: int,
                          label_2_index: int) -> np.ndarray:
    """
    Computes the binary compatibility function values for two given detected labels for all
    possible values of assigned labels f_i and f_j.

    Args:
        label_1 (DetectedLabel):    A detected label.
        label_2 (DetectedLabel):    A detected label.
        label_1_index (int):        The index (within the LabelStructure) of the first label.
                                        Example, index of 'C' is 3.
        label_2_index (int):        The index of the second label.

    Returns:
        result (np.ndarray):    The values of the binary compatibility function for each of the
                                    possible pairs of input values (f_i, f_j).
                                    -> Shape = (2, 2)
    """

    assert label_1.box is not None and label_2.box is not None
    assert label_1.text is not None and label_2.text is not None

    ratio_areas: float = box.area(label_1.box) / box.area(label_2.box)

    label_1_pos: box.Point = box.get_center(label_1.box)
    label_2_pos: box.Point = box.get_center(label_2.box)

    delta_x: int = label_2_pos.x - label_1_pos.x
    delta_y: int = label_2_pos.y - label_1_pos.y

    text_labels_relationship: bool

    result: np.ndarray = np.zeros(shape=(2, 2))

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

    result[1, 1] = GAMMA * ratio_areas + DELTA * int(text_labels_relationship)
    result[0, 0] = 1 - result[1, 1]
    result[0, 1] = 1 - result[1, 1]
    result[1, 0] = 1 - result[1, 1]

    return result


def _is_label_in_prior_zone(label_1: DetectedLabel,
                            label_2: DetectedLabel) -> bool:
    """
    Args:
        label_1 (DetectedLabel):    A detected label.
        label_2 (DetectedLabel):    A detected label.

    Returns:
        is_in_prior_zone (bool):    True if label_2 is in the prior zone of label_1.
    """

    label_1_pos: box.Point = box.get_center(label_1.box)
    label_2_pos: box.Point = box.get_center(label_2.box)

    is_at_left = label_2_pos.x < label_1_pos.x

    is_above = label_2_pos.y < label_1_pos.y

    return is_at_left or is_above


def _build_neighbors(label_list: List[DetectedLabel]) -> np.ndarray:
    """
    Builds the adjacency matrix.
    adjacency_matrix[i, j] = 1 if and only if labels i and j are neighbors.

    Args:
        label_list (List[DetectedLabel]):   List of detected labels.

    Returns:
        adjacency_matrix (np.ndarray):  The asdjacency (symmetric) matrix of the underlying graph.
                                            -> Shape = (num_labels, num_labels)
    """
    num_labels: int = len(label_list)

    adjacency_matrix: np.ndarray = np.zeros(shape=(num_labels, num_labels))
    for i, label in enumerate(label_list):

        assert label.box is not None

        label_x_left, label_y_top, label_x_right, label_y_bot = label.box

        for j, potential_neighbor in enumerate(label_list):

            if potential_neighbor == label:
                continue

            assert potential_neighbor.box is not None

            potential_neighbor_center: box.Point = box.get_center(potential_neighbor.box)

            is_aligned_horizontally: bool = potential_neighbor_center.x >= label_x_left \
                and potential_neighbor_center.x <= label_x_right

            is_aligned_vertically: bool = potential_neighbor_center.y >= label_y_top \
                and potential_neighbor_center.y <= label_y_bot

            if is_aligned_horizontally or is_aligned_vertically:
                adjacency_matrix[i, j] = 1

    assert all(adjacency_matrix[i, j] == adjacency_matrix[j, i]
               for i in range(num_labels)
               for j in range(num_labels))

    return adjacency_matrix


def _precompute_compatibility_functions(label_list: List[DetectedLabel]
                                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute the compatibility function values.

    Args:
        label_list (List[DetectedLabel]):   The list of originally detected labels.

    Returns:
        unary_compatibility_values (np.ndarray):    The values of the unary compatibility function
                                                        for both of the possible input
                                                        values f_i for each of the detected
                                                        labels f_i.
                                                        -> Shape = (num_labels, 2)

        binary_compatibility_values (np.ndarray):   The values of the binary compatibility function
                                                        for each of the possible pairs of input
                                                        values (f_i, f_j) for each pair of labels
                                                        (i, j).
                                                        -> Shape = (num_labels, num_labels, 2, 2)

    """
    label_text_list: List[str] = [label.text for label in label_list]


    label_structure_type: LabelStructureEnum = LabelStructure.from_labels_list(
    labels_list=label_text_list).labels_type

    num_labels: int = len(label_list)

    unary_compatibility_values: np.ndarray = np.zeros(shape=(num_labels, 2))
    binary_compatibility_values: np.ndarray = np.zeros(shape=(num_labels, num_labels, 2, 2))

    for i, label_i in enumerate(label_list):

        num_labels_in_prior_zone: int = 0
        label_i_index: int = LABEL_INDEX[label_structure_type](label_i.text)

        for j, label_j in enumerate(label_list):

            num_labels_in_prior_zone += int(_is_label_in_prior_zone(label_1=label_i,
                                                                    label_2=label_j))


            label_j_index: int = LABEL_INDEX[label_structure_type](label_j.text)
            binary_compatibility_values[i, j] = _binary_compatibility(label_1=label_i,
                                                                      label_2=label_j,
                                                                      label_1_index=label_i_index,
                                                                      label_2_index=label_j_index)


        unary_compatibility_values[i] = _unary_compatibility(
            label=label_i,
            label_index=label_i_index,
            number_of_labels_in_prior_zone=num_labels_in_prior_zone)


    return unary_compatibility_values, binary_compatibility_values


def _belief_propagation(label_list: List[DetectedLabel]) -> np.ndarray:
    """
    Apply the belief propagation algorithm to compute the beliefs.

    Args:
        label_list (List[DetectedLabel]):   The list of detected labels.

    Returns:
        beliefs (np.ndarray):   The value of the beliefs for each label.
                                    Shape: (num_labels, 2)
    """
    # Compute the adjacency matrix
    adjacency_matrix: np.ndarray = _build_neighbors(label_list=label_list)

    unary_compatibility_values, binary_compatibility_values = _precompute_compatibility_functions(
        label_list=label_list)

    num_labels: int = len(label_list)

    # Compute the messages iteratively.

    # messages[i, j][f_j] = m_ij(f_j)
    messages = np.zeros(shape=(num_labels, num_labels, 2))

    # TODO set a convergence criterion (what is a 'label flip' ?)
    while True:
        for i in range(num_labels):
            for j in range(num_labels):
                if i == j:
                    continue

                if adjacency_matrix[i, j] == 0:
                    continue

                # m_ij(f_j) = max f_i [r_i(f_i) r_ij(f_i, f_j) max k m_ki(f_i)]
                messages[i, j] = [
                    np.max([
                       unary_compatibility_values[i][f_i] \
                           * binary_compatibility_values[i, j][f_i, f_j] \
                           * np.max([
                               messages[k, i][f_i]
                               for k in range(num_labels)
                               if k != j and adjacency_matrix[k, i] == 1]
                            )
                       for f_i in (0, 1)]
                    )
                    for f_j in (0, 1)]


    beliefs: np.ndarray = np.zeros(shape=(num_labels))

    for i in range(num_labels):
        beliefs[i] = [unary_compatibility_values[i][f_i] * np.max([
                            messages[j, i][f_i]
                            for j in range(num_labels)
                            if adjacency_matrix[j, i] == 1])
                       for f_i in (0, 1)]

    return beliefs



def filter_labels(label_list: List[DetectedLabel],
                  belief_threshold = 0.8) -> List[DetectedLabel]:
    """
    Filter false positive labels previously detected in the image.

    Args:
        label_list (List[DetectedLabel]):   A list of detected labels.

    Returns:
        filtered_labels (List[DetectedLabel]):  The list of filtered labels.
    """
    beliefs: np.ndarray = _belief_propagation(label_list=label_list)

    filtered_labels: List[DetectedLabel] = [
        label
        for i, label in enumerate(label_list)
        if beliefs[i] > belief_threshold
    ]

    return filtered_labels
