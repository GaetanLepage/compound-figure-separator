#!/usr/bin/env python3

"""
TODO
"""

import sys
import argparse

sys.path.append(".")

from panel_seg.io.figure_generators import image_clef_xml_figure_generator
from panel_seg.io.export import export_figures_to_csv


def parse_args(args):
    """
    Parse the arguments from the command line.

    Args:
        args: The arguments from the command line call.

    Returns:
        Populated namespace
    """
    parser = argparse.ArgumentParser(description='Convert annotations from an ImageCLEF xml"\
                                        " annotation file to a CSV annotations file.')

    parser.add_argument('--annotation_xml',
                        help='The path to the xml annotation file.',
                        default='data/imageCLEF/test/FigureSeparationTest2016GT.xml',
                        type=str)

    parser.add_argument('--image_directory_path',
                        help='The path to the directory whre the images are stored.',
                        default='data/imageCLEF/test/FigureSeparationTest2016/',
                        type=str)

    parser.add_argument('--output_csv',
                        help='The path of the csv file to which annotations have to be exported.',
                        default='data/imageCLEF/test/test.csv',
                        type=str)

    parser.add_argument('--individual_csv',
                        help='Also export the annotations to a single csv file.',
                        action='store_true')

    parser.add_argument('--individual_export_csv_directory',
                        help='The path of the directory whete to store the individual csv"\
                                " annotation files.',
                        default='data/imageCLEF/test/test.csv',
                        type=str)

    return parser.parse_args(args)


def main(args=None):
    """
    Load figures from ImageCLEF xml annotation files and export them to csv.
    """

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Create the figure generator handling xml annotation files
    figure_generator = image_clef_xml_figure_generator(
        xml_annotation_file_path=args.annotation_xml,
        image_directory_path=args.image_directory_path)

    # Export figures to csv
    export_figures_to_csv(
        figure_generator=figure_generator,
        output_csv_file=args.output_csv,
        individual_export=args.individual_csv,
        individual_export_csv_directory=args.individual_export_csv_directory)


if __name__ == '__main__':
    main()
