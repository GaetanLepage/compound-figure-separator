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


###############################################################################
Functions to merge image label and text label information to filter detections.
"""

from typing import List

from ..utils.figure.label import (LabelStructure)


def label_filtering(text_labels: List[str],
                    image_labels: List[str] = []) -> LabelStructure:
    """
    Cross the image-based and text-based label detections.

    Args:
        text_labels (List[str]):    List of labels extracted from the caption text.
        image_labels (List[str]):   List of labels extracted from the image. (optional)

    Returns:
        label_structure (LabelStructure):   TODO.
    """

    # Merge the two lists of labels.
    # Note: The duplicates are not removed on purpose. The idea is that each detection
    # vote for the most likely label structure.
    labels: List[str] = text_labels + image_labels

    label_structure: LabelStructure = LabelStructure.from_labels_list(labels_list=labels)

    return label_structure
