#!/usr/bin/env python3
"""
Script to visualize Zou's data set by displaying the images along with the corresponding
bounding boxes and labels.
"""

import sys
from argparse import ArgumentParser

from typing import List

sys.path.append(".")

from panel_seg.data.figure_generators import iphotodraw_xml_figure_generator
from panel_seg.data.figure_viewer import parse_viewer_args, view_data_set

def parse_args(args: List[str]) -> ArgumentParser:
    """
    Parse the arguments from the command line.

    Args:
        args (List[str]): The arguments from the command line call.

    Returns:
        (ArgumentParser): Populated namespace.
    """
    parser = ArgumentParser(description="Preview all the figures from an iPhotoDraw data set.")

    parser.add_argument('--eval_list_txt',
                        help='The path to the txt file listing the images.',
                        default='data/zou/eval.txt',
                        type=str)

    parser.add_argument('--image_directory_path',
                        help='The path to the directory whre the images are stored.',
                        default=None,
                        type=str)

    parser = parse_viewer_args(parser)

    return parser.parse_args(args)


def main(args: List[str] = None):
    """
    Launch previsualization of Zou's data set.

    Args:
        args (List[str]): Arguments from the command line.
    """

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Create the figure generator handling iPhotoDraw annotation files.
    figure_generator = iphotodraw_xml_figure_generator(
        eval_list_txt=args.eval_list_txt,
        image_directory_path=args.image_directory_path)

    # Preview the data set.
    view_data_set(figure_generator=figure_generator,
                  mode=args.mode,
                  delay=args.delay,
                  window_name="PanelSeg data preview")


if __name__ == '__main__':
    main()
