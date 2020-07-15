#!/usr/bin/env python3

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

Collaborator:   Niccolò Marini (niccolo.marini@hevs.ch)


#############################################################################
Script to evaluate label recognition predictions from a JSON annotation file.
"""

import sys
import os
from argparse import ArgumentParser

from typing import List

import compfigsep

sys.path.append('.')

MODULE_DIR = os.path.dirname(compfigsep.__file__)

from compfigsep.data.figure_generators import JsonFigureGenerator, get_most_recent_json
from compfigsep.label_recognition.evaluate import evaluate_detections


def parse_args(args: List[str]) -> ArgumentParser:
    """
    Parse the arguments from the command line.

    Args:
        args (List[str]):   The arguments from the command line call.

    Returns:
        parser (ArgumentParser):    Populated namespace.
    """
    parser = ArgumentParser(description="Evaluate label recognition detections.")

    # TODO check default path
    parser.add_argument('--json',
                        help="The path to the json annotation file.",
                        default=get_most_recent_json(
                            folder_path=os.path.join(
                                MODULE_DIR,
                                'label_recognition/output/')),
                        type=str)

    return parser.parse_args(args)


def main(args: List[str] = None):
    """
    Launch evaluation of the label recognition task on a JSON data set.

    Args:
        args (List[str]):   Arguments from the command line.
    """

    # Parse arguments.
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Create the figure generator handling JSON annotation files.
    figure_generator = JsonFigureGenerator(
        json_annotation_file_path=args.json)

    # Evaluate the data set.
    evaluate_detections(figure_generator=figure_generator)


if __name__ == '__main__':
    main()