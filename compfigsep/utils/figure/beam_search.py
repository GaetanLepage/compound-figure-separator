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


####################################################
Beam search algorithm for mapping panels and labels.
"""

from typing import NamedTuple, Sequence

from ...utils import box
from ...utils.figure.panel import Panel, DetectedPanel
from ...utils.figure.label import Label, DetectedLabel
from ...utils.figure.sub_figure import SubFigure, DetectedSubFigure


class Path(NamedTuple):
    """
    Attributes:
        overall_distance (float):   TODO
        label_indexes (list[int]):  TODO
    """

    overall_distance: float
    label_indexes: list[int]


def _compute_panel_label_distances(
        panels: Sequence[Panel],
        labels: Sequence[Label]
) -> list[list[float]]:
    """
    Compute distances between each label and each panel.

    Args:
        panels (list[Panel]):   The list of panels.
        labels (list[Label]):   The list of labels.

    Returns:
        distance_matrix (list[list[float]]):    The matrix containing the distances.
    """
    # calculate distance from panels to labels
    # distance_matrix[i, j] = dist(Panel_i, Label_j)
    distance_matrix: list[list[float]] = []
    for panel in panels:
        dist_from_current_panel_to_labels: list[float] = []

        panel_box_l: int = panel.box[0]
        panel_box_r: int = panel.box[2]
        panel_box_t: int = panel.box[1]
        panel_box_b: int = panel.box[3]

        for label in labels:
            if label.box is None:
                continue

            label_box_l: int = label.box[0]
            label_box_r: int = label.box[2]
            label_box_t: int = label.box[1]
            label_box_b: int = label.box[3]

            if label_box_r < panel_box_l:
                h_dist: int = panel_box_l - label_box_r
            elif label_box_l > panel_box_r:
                h_dist = label_box_l - panel_box_r
            else:
                h_dist = 0

            if label_box_b < panel_box_t:
                v_dist: int = panel_box_t - label_box_b
            elif label_box_t > panel_box_b:
                v_dist = label_box_t - panel_box_b
            else:
                v_dist = 0

            distance: float = h_dist + v_dist
            dist_from_current_panel_to_labels.append(distance)

        distance_matrix.append(dist_from_current_panel_to_labels)

    return distance_matrix


def assign_labels_to_panels(
        panels: Sequence[Panel],
        labels: Sequence[Label],
        are_detections: bool,
        beam_length: int = 100
) -> list[SubFigure]:
    """
    Use beam search to assign labels to panels according to the overall distance
    Assign labels.label_rect to panels.label_rect.
    panels and labels must have the same length.
    TODO: solve the problem when length differs.

    Args:
        panels (list[Panel]):   list of panels.
        labels (list[Label]):   list of labels.
        are_detections (bool):  If the given panels and labels are detections.
                                    If False, it means that we are matching ground truth
                                    annotations.
        beam_length (int):      The number of top paths kept in each column.

    Returns:
        subfigures (list[SubFigure]):   The list of resulting subfigures (i.e. a list of panel-label
                                            pairs).
    """

    num_panels: int = len(panels)

    if num_panels == 0:
        return []

    num_labels: int = len(labels)

    if num_labels == 0:
        if are_detections:
            detected_subfigures: list[DetectedSubFigure] = []
            for detected_panel in panels:
                assert isinstance(detected_panel, DetectedPanel)
                detected_subfigures.append(
                    DetectedSubFigure(panel=detected_panel)
                )

        return [
            SubFigure(panel=gt_panel)
            for gt_panel
            in panels
        ]

    print(f"num_panels = {num_panels}")
    print(f"num_labels = {num_labels}")

    if are_detections:

        assert all(isinstance(panel, DetectedPanel) for panel in panels)

        # Sort the panels according to their detection score
        panels.sort(key=lambda detected_panel: detected_panel.detection_score)  # type: ignore

        assert all(isinstance(label, DetectedLabel) for label in labels)

        # Sort the panels according to their detection score
        labels.sort(key=lambda detected_label: detected_label.detection_score)  # type: ignore

    # Compute the distance matrix.
    distance_matrix: list[list[float]] = _compute_panel_label_distances(
        panels=panels,
        labels=labels
    )

    # Beam search

    all_paths: list[list[Path]] = []

    for panel_idx, panel in enumerate(panels):

        panel_paths: list[Path] = []

        panel_width, panel_height = box.get_width_and_height(box=panel.box)

        # Initialisation
        if panel_idx == 0:
            for label_idx in range(num_labels):
                dist: float = distance_matrix[panel_idx][label_idx]
                # we do not allow the distance to be larger than the 1/2 of panel side
                if dist > panel_width / 2 or dist > panel_height / 2:
                    continue

                label_indexes: list[int] = [label_idx]
                path: Path = Path(
                    overall_distance=dist,
                    label_indexes=label_indexes
                )

                panel_paths.append(path)

            # Manually add the path corresponding to the association of this panel with no label.
            panel_paths.append(
                Path(
                    overall_distance=0,
                    label_indexes=[-1]
                )
            )

        # Exploring the graph.
        else:
            prev_paths: list[Path] = all_paths[panel_idx - 1]

            for prev_path in prev_paths:

                prev_dist = prev_path.overall_distance
                prev_label_indexes = prev_path.label_indexes

                for label_idx in range(num_labels):
                    if label_idx in prev_label_indexes:
                        # # We allow a label assigned to one panel only
                        continue

                    dist = distance_matrix[panel_idx][label_idx] + prev_dist

                    # we do not allow the distance to be larger than the 1/2 of panel side
                    if dist > panel_width / 2 or dist > panel_height / 2:
                        continue

                    # I think I did this to copy the `prev_label_indexes` list
                    # Maybe use `.copy()`...
                    label_indexes = list(prev_label_indexes)
                    label_indexes.append(label_idx)
                    path = Path(
                        overall_distance=dist,
                        label_indexes=label_indexes
                    )
                    panel_paths.append(path)

                # Manually add the path corresponding to the association of this panel with no
                # label.
                label_indexes = list(prev_label_indexes)
                label_indexes.append(-1)
                panel_paths.append(
                    Path(
                        overall_distance=prev_dist,
                        label_indexes=label_indexes
                    )
                )

        # sort item_pairs
        panel_paths.sort(key=lambda path: path.overall_distance)
        # keep only at most beam_length paths
        if len(panel_paths) > beam_length:
            panel_paths = panel_paths[:beam_length]

        all_paths.append(panel_paths)

        # TODO remove
        print("panel index:", panel_idx)
        print("all_item_pairs:", all_paths)

    # check the last column of paths (which corresponds to full paths)
    # all_item_pairs[-1] :                  last "layer" (complete paths)
    # all_item_pairs[-1][0] :               pair which has the shortest path (it was sorted)
    # all_item_pairs[-1][0].label_indexes : the complete path from this pair
    # TODO remove
    # print(all_paths[-1])
    best_path = all_paths[-1][0].label_indexes

    subfigures: list[SubFigure] = []

    for panel_index, panel in enumerate(panels):
        # panel.add_label_info(label=labels[best_path[panel_index]])
        matched_label: Label = labels[best_path[panel_index]]

        if are_detections:
            assert isinstance(panel, DetectedPanel)
            assert isinstance(matched_label, DetectedLabel)
            subfigure: SubFigure = DetectedSubFigure(
                panel=panel,
                label=matched_label
            )
        else:
            subfigure = SubFigure(
                panel=panel,
                label=matched_label
            )

        subfigures.append(subfigure)

    return subfigures
