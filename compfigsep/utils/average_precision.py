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


##########################################
Function to compute the average precision.
"""

import numpy as np

def compute_average_precision(recall: np.ndarray,
                              precision: np.ndarray) -> float:
    """
    Compute the average precision as the area under the precision/recall curve.
    This computation method is the one used for PascalVOC 2012 challenge.

    Args:
        recall (np.array[float]):       The list of recall values.
        precision (np.array[float]):    TODO.

    Returns:
        average_precision (float):  The resulting average precision.
    """

    assert len(recall) == len(precision), "`recall` and `precision` should have the same"\
                                          f" length.\n\tlen(recall) = {len(recall)}"\
                                          f"\n\tlen(precision) = {len(precision)}"\

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    # Make the precision monotonically decreasing (goes from the end to the beginning)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])


    # To calculate area under PR curve, look for points
    # where X axis (recall) changes value.
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum (delta recall) * prec.
    average_precision = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return average_precision
