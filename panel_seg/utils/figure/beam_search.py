"""
Beam search algorithm for mapping panels and labels.
"""

import math
from typing import List

from panel_seg.utils import box
from panel_seg.utils.figure.panel import Panel


def compute_panel_label_distances(panels: List[Panel], labels: List[Panel]):
    """
    Compute distances between each label and each panel.

    Args:
        panels: The list of Panel objects with
        labels:

    Returns:
        The matrix containing the distances.
    """
    # calculate distance from panels to labels
    distances = []
    for panel in panels:
        dists = []
        panel_rect = panel.panel_rect
        panel_center = box.get_center(panel_rect)

        for label in labels:
            label_rect = label.label_rect
            label_center = box.get_center(label_rect)

            distance = math.hypot(
                panel_center[0] - label_center[0],
                panel_center[1] - label_center[1])
            dists.append(distance)
        distances.append(dists)

    return distances


def assign_labels_to_panels(panels: List[Panel],
                            labels: List[Panel],
                            beam_length: int = 100):
    """
    Use beam search to assign labels to panels according to the overall distance
    Assign labels.label_rect to panels.label_rect
    panels and labels must have the same length

    Args:
        panels (List[Panel]):   Panels having the same label character.
        labels (List[Panel]):   Labels having the same label character.
        beam_length (int):      TODO
    """
    # TODO remove
    print("###########")

    # Compute the distance matrix.
    distances = compute_panel_label_distances(panels, labels)

    # Beam search

    # a `pair` represents a path (overall_distance, label_indexes)
    all_item_pairs = []
    for panel_idx, panel in enumerate(panels):
        item_pairs = []

        # Initialisation
        if panel_idx == 0:
            for label_idx in range(len(labels)):
                dist = distances[panel_idx][label_idx]
                label_indexes = [label_idx]
                item_pair = (dist, label_indexes)
                item_pairs.append(item_pair)

        # Exploring the graph
        else:
            prev_item_pairs = all_item_pairs[panel_idx - 1]
            for prev_item_pair in prev_item_pairs:

                prev_dist, prev_label_indexes = prev_item_pair

                for label_idx in range(len(labels)):
                    if label_idx in prev_label_indexes:
                        # We allow a label assigned to one panel only
                        continue
                    dist = distances[panel_idx][label_idx] + prev_dist
                    label_indexes = list(prev_label_indexes)
                    label_indexes.append(label_idx)
                    item_pair = (dist, label_indexes)
                    item_pairs.append(item_pair)

        # sort item_pairs
        item_pairs.sort(key=lambda pair: pair[0])
        # keep only at most beam_length item pairs
        if len(item_pairs) > 100:
            item_pairs = item_pairs[:beam_length]

        all_item_pairs.append(item_pairs)

        # TODO remove
        print("panel index:", panel_idx)
        print("all_item_pairs:", all_item_pairs)

    # check the last item_pairs
    # print(all_item_pairs)
    # all_item_pairs[-1] : last "layer" (complete paths)
    # all_item_pairs[-1][0] : pair which has the shortest path (it was sorted)
    # all_item_pairs[-1][0][1] : the complete path from this pair
    best_path = all_item_pairs[-1][0][1]
    for panel_index, panel in enumerate(panels):
        panel.add_label_info(label=labels[best_path[panel_index]])
