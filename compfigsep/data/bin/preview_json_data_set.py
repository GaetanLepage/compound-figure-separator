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


##################################################################################################
Script to visualize a JSON data set by displaying the images along with the corresponding bounding
boxes.
"""

import sys
from argparse import ArgumentParser, Namespace

from typing import List

from compfigsep.data.figure_generators import JsonFigureGenerator
from compfigsep.data.figure_viewer import add_viewer_args, view_data_set

sys.path.append('.')

def parse_args(args: List[str]) -> Namespace:
    """
    Parse the arguments from the command line.

    Args:
        args (List[str]):   The arguments from the command line call.

    Returns:
        namespace (Namespace):  Populated namespace.
    """
    parser: ArgumentParser = ArgumentParser(
        description="Preview all the figures from an ImageCLEF data set.")

    # TODO check default path
    parser.add_argument('--json',
                        help="The path to the json annotation file.",
                        default="experiments/panel_segmentation/test.json",
                        type=str)

    add_viewer_args(parser)

    return parser.parse_args(args)


def main(args: List[str] = None):
    """
    Launch previsualization of a JSON data set.

    Args:
        args (List[str]):   Arguments from the command line.
    """

    # Parse arguments.
    if args is None:
        args = sys.argv[1:]
    parsed_args: Namespace = parse_args(args)

    # Create the figure generator handling ImageCLEF xml annotation files.
    figure_generator: JsonFigureGenerator = JsonFigureGenerator(
        json_path=parsed_args.json)

    # Preview the data set.
    view_data_set(figure_generator=figure_generator,
                  mode=parsed_args.mode,
                  save_preview=parsed_args.save_preview,
                  preview_folder=parsed_args.json.replace('.json', '_preview/'),
                  delay=parsed_args.delay,
                  window_name="JSON data preview")


if __name__ == '__main__':
    main()
