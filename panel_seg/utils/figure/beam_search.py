"""
Miscellaneous functions for figures.
"""

import math
from typing import List

from .panel import Panel


def compute_panel_label_distances(panels, labels):
    """
    Compute distances between each label and each panel.

    Args:
        panels: the list of Panel objects.
        labels:

    Returns:
        The matrix containing the distances.
    """
    # calculate distance from panels to labels
    distances = []
    for panel in panels:
        dists = []
        panel_rect = panel.panel_rect
        panel_center = [
            (panel_rect[0] + panel_rect[2])/2.0,
            (panel_rect[1] + panel_rect[3])/2.0]

        for label in labels:
            label_rect = label.label_rect
            label_center = [
                (label_rect[0] + label_rect[2])/2.0,
                (label_rect[1] + label_rect[3])/2.0]

            distance = math.hypot(
                panel_center[0] - label_center[0],
                panel_center[1] - label_center[1])
            dists.append(distance)
        distances.append(dists)

    return distances


def assign_labels_to_panels(panels: List[Panel],
                            labels: List[str],
                            beam_length: int = 100):
    """
    Use beam search to assign labels to panels according to the overall distance
    Assign labels.label_rect to panels.label_rect
    panels and labels must have the same length

    Args:
        panels: panels having the same label character
        labels: labels having the same label character
    """

    distances = compute_panel_label_distances(panels, labels)

    # Beam search

    # in the format (overall_distance, label_indexes)
    all_item_pairs = []
    for panel_i, panel in enumerate(panels):
        item_pairs = []
        if panel_i == 0:
            for label_index in range(len(labels)):
                dist = distances[panel_i][label_index]
                label_indexes = [label_index]
                item_pair = [dist, label_indexes]
                item_pairs.append(item_pair)
        else:
            prev_item_pairs = all_item_pairs[panel_i - 1]
            for prev_item_pair in prev_item_pairs:
                prev_label_indexes = prev_item_pair[1]
                prev_dist = prev_item_pair[0]
                for label_index in range(len(labels)):
                    if label_index in prev_label_indexes:
                        # We allow a label assigned to one panel only
                        continue
                    dist = distances[panel_i][label_index] + prev_dist
                    label_indexes = list(prev_item_pair[1])
                    label_indexes.append(label_index)
                    item_pair = [dist, label_indexes]
                    item_pairs.append(item_pair)

        # sort item_pairs
        item_pairs.sort(key=lambda pair: pair[0])
        # keep only at most beam_length item pairs
        if len(item_pairs) > 100:
            item_pairs = item_pairs[:beam_length]

        all_item_pairs.append(item_pairs)

    # check the last item_pairs
    best_path = all_item_pairs[-1][0][1]
    for panel_index, panel in enumerate(panels):
        panel.label_rect = labels[best_path[panel_index]].label_rect
