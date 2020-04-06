#! /usr/bin/env python3

"""
Script used to evaluate the performance of the splitting algorithm
with respect to the ImageCLEF metric.
"""

def load_ground_truth_from_csv(self, path_to_csv):
    """
    TODO
    """


def eval_splitting(
        ground_truth_sub_figures,
        candidate_sub_figures):
    """
    Compute some metrics by comparing estimated (candidates) sub figures
    and the ground trouth ones:
        *

    Args:
        ground_truth_sub_figures: TODO
        candidate_sub_figures: TODO

    Returns:
        TODO
    """

    assert len(ground_truth_sub_figures) == len(candidate_sub_figures),\
            "The two given sets of figures do not have the same size :" \
            "\n\tGround truth : {} subfigures\n\tCandidates : {} subfigures"

    # Care order of looping
