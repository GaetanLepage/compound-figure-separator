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


#############################################################################
Script to evaluate label recognition predictions from a JSON annotation file.
"""

import sys
import os
from argparse import ArgumentParser, Namespace

import compfigsep
from compfigsep.data.figure_generators import JsonFigureGenerator, add_json_arg
from compfigsep.label_recognition.evaluate import evaluate_detections

sys.path.append('.')

MODULE_DIR: str = os.path.dirname(compfigsep.__file__)


def parse_args(args: list[str]) -> Namespace:
    """
    Parse the arguments from the command line.

    Args:
        args (list[str]):   The arguments from the command line call.

    Returns:
        namespace (Namespace):  Populated namespace.
    """
    parser = ArgumentParser(description="Evaluate label recognition detections.")

    add_json_arg(
        parser=parser,
        folder_default_path='label_recognition/output/'
    )

    return parser.parse_args(args)


def main(args: list[str] = None):
    """
    Launch evaluation of the label recognition task on a JSON data set.

    Args:
        args (list[str]):   Arguments from the command line.
    """

    # Parse arguments.
    if args is None:
        args = sys.argv[1:]
    parsed_args = parse_args(args)

    # Create the figure generator handling JSON annotation files.
    figure_generator = JsonFigureGenerator(json_path=parsed_args.json)

    # Evaluate the data set.
    evaluate_detections(figure_generator=figure_generator)


if __name__ == '__main__':
    main()
