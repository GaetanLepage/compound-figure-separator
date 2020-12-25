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


#######################################################################
Script to visualize a "csv annotated" data set by displaying the images
along with the corresponding bounding boxes and labels.
"""

import sys
from argparse import ArgumentParser, Namespace

from typing import List

from compfigsep.data.figure_generators import GlobalCsvFigureGenerator
from compfigsep.data.figure_viewer import add_viewer_args, view_data_set

sys.path.append('.')


def parse_args(args: List[str]) -> Namespace:
    """
    Parse the arguments from the command line.

    Args:
        args (List[str]):   The arguments from the command line call.

    Returns:
        parser (Namespace): Populated namespace.
    """
    parser: ArgumentParser = ArgumentParser(description="Preview all the figures from a data set"\
                                                        " represented by a csv annotation file.")

    parser.add_argument('--annotation_csv',
                        help='The path to the csv annotation file.',
                        type=str)

    add_viewer_args(parser)

    return parser.parse_args(args)


def main(args: List[str] = None) -> None:
    """
    Launch previsualization of a csv data set.

    Args:
        args (List[str]):   Arguments from the command line.
    """

    # Parse arguments.
    if args is None:
        args = sys.argv[1:]
    parsed_args: Namespace = parse_args(args)

    # Create the figure generator handling a csv annotation file.
    figure_generator = GlobalCsvFigureGenerator(
        csv_annotation_file_path=parsed_args.annotation_csv)

    # Preview the data set.
    view_data_set(figure_generator=figure_generator,
                  mode=parsed_args.mode,
                  delay=parsed_args.delay,
                  window_name="CSV data preview")


if __name__ == '__main__':
    main()
