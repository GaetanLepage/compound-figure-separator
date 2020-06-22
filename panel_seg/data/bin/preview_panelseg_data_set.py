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


########################################################################################
Script to visualize Zou's data set by displaying the images along with the corresponding
bounding boxes and labels.
"""

import sys
import os
from argparse import ArgumentParser

from typing import List

from ..figure_generators import IphotodrawXmlFigureGenerator
from ..figure_viewer import parse_viewer_args, view_data_set

sys.path.append(".")

def parse_args(args: List[str]) -> ArgumentParser:
    """
    Parse the arguments from the command line.

    Args:
        args (List[str]):   The arguments from the command line call.

    Returns:
        parser (ArgumentParser):    Populated namespace.
    """
    parser = ArgumentParser(description="Preview all the figures from an iPhotoDraw data set.")

    parser.add_argument('--eval_list_txt',
                        help="The path to the txt file listing the images.",
                        default="data/zou/eval.txt",
                        type=str)

    parser.add_argument('--image_directory_path',
                        help="The path to the directory whre the images are stored.",
                        default=None,
                        type=str)

    parser = parse_viewer_args(parser)

    return parser.parse_args(args)


def main(args: List[str] = None):
    """
    Launch previsualization of Zou's data set.

    Args:
        args (List[str]):   Arguments from the command line.
    """

    # Parse arguments.
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Create the figure generator handling iPhotoDraw annotation files.
    figure_generator = IphotodrawXmlFigureGenerator(
        eval_list_txt=args.eval_list_txt,
        image_directory_path=args.image_directory_path)

    if args.save_preview:
        if args.eval_list_txt is not None:
            preview_folder = os.path.dirname(args.eval_list_txt)
        else:
            preview_folder = args.image_directory_path
        preview_folder = os.path.join(preview_folder, 'preview')
    else:
        preview_folder = ""


    # Preview the data set.
    view_data_set(figure_generator=figure_generator(),
                  mode=args.mode,
                  save_preview=args.save_preview,
                  preview_folder=preview_folder,
                  delay=args.delay,
                  window_name="PanelSeg data preview")


if __name__ == '__main__':
    main()
