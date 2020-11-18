#!/usr/bin/env python3

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


##############################################################################
Script to evaluate panel segmentation predictions from a JSON annotation file.
"""

import sys
from argparse import ArgumentParser, Namespace

from typing import List


sys.path.append('.')


from compfigsep.data.figure_generators import JsonFigureGenerator, add_json_arg

from compfigsep.panel_segmentation.evaluate import evaluate_detections


def parse_args(args: List[str]) -> Namespace:
    """
    Parse the arguments from the command line.

    Args:
        args (List[str]):   The arguments from the command line call.

    Returns:
        parser (Namespace):    Populated namespace.
    """
    parser = ArgumentParser(description="Evaluate panel segmentation detections.")

    add_json_arg(parser=parser,
                 folder_default_relative_path='panel_segmentation/output/')

    return parser.parse_args(args)


def main(args: List[str] = None):
    """
    Launch evaluation of the panel segmentation task on a JSON data set.

    Args:
        args (List[str]):   Arguments from the command line.
    """

    # Parse arguments.
    if args is None:
        args = sys.argv[1:]
    parsed_args: Namespace = parse_args(args)

    # Create the figure generator handling JSON annotation files.
    figure_generator = JsonFigureGenerator(
        json_path=parsed_args.json)

    # Evaluate the data set.
    evaluate_detections(figure_generator=figure_generator)


if __name__ == '__main__':
    main()
