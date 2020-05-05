"""
Script to visualize a "csv annotated" data set by displaying the images
along with the corresponding bounding boxes and labels.
"""

import sys
import argparse

sys.path.append('.')

from panel_seg.io.figure_generators import global_csv_figure_generator
from panel_seg.io.figure_viewer import view_data_set


def parse_args(args):
    """
    Parse the arguments from the command line.

    Args:
        args: The arguments from the command line call.

    Returns:
        Populated namespace
    """
    parser = argparse.ArgumentParser(
        description='Preview all the figures from a data set represented by a csv annotation file.')

    parser.add_argument('--annotation_csv',
                        help='The path to the csv annotation file.',
                        type=str)

    return parser.parse_args(args)


def main(args=None):
    """
    Launch previsualization of a csv data set.
    """

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    figure_generator = global_csv_figure_generator(
        csv_annotation_file_path=args.annotation_csv)

    view_data_set(
        figure_generator=figure_generator,
        delay=100,
        window_name="CSV data preview")


if __name__ == '__main__':
    main()
