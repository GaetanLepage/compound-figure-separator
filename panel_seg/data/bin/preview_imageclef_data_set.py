#!/usr/bin/env python3
"""
Script to visualize the ImageCLEF data set by displaying the images along with the corresponding
bounding boxes.
"""

import sys
import argparse

sys.path.append('.')

from panel_seg.io.figure_generators import image_clef_xml_figure_generator
from panel_seg.io.figure_viewer import parse_viewer_args, view_data_set


def parse_args(args):
    """
    Parse the arguments from the command line.

    Args:
        args: The arguments from the command line call.

    Returns:
        Populated namespace
    """
    parser = argparse.ArgumentParser(
        description='Preview all the figures from an ImageCLEF data set.')

    parser.add_argument('--annotation_xml',
                        help='The path to the xml annotation file.',
                        default='data/ImageCLEF/test/FigureSeparationTest2016GT.xml',
                        type=str)

    parser.add_argument('--image_directory_path',
                        help='The path to the directory whre the images are stored.',
                        default='data/ImageCLEF/test/FigureSeparationTest2016/',
                        type=str)

    parser = parse_viewer_args(parser)

    return parser.parse_args(args)


def main(args=None):
    """
    Launch previsualization of ImageCLEF data set.
    """

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    figure_generator = image_clef_xml_figure_generator(
        xml_annotation_file_path=args.annotation_xml,
        image_directory_path=args.image_directory_path)

    view_data_set(figure_generator=figure_generator,
                  mode=args.mode,
                  delay=args.delay,
                  window_name="ImageCLEF data preview")


if __name__ == '__main__':
    main()
