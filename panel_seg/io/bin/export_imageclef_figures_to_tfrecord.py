#!/usr/bin/env python3

"""
Script to export the ImageCLEF dataset to a tfrecord file compatible with
Tensorflow's Object Detection API.
(https://github.com/tensorflow/models/tree/master/official/vision/detection)
"""

import sys
import argparse

sys.path.append(".")

from panel_seg.io.figure_generators import image_clef_xml_figure_generator
from panel_seg.io.export import export_figures_to_tf_record


def parse_args(args):
    """
    Parse the arguments from the command line.

    Args:
        args: The arguments from the command line call.

    Returns:
        Populated namespace
    """
    parser = argparse.ArgumentParser(description="Convert the ImageCLEF dataset to a"\
                                                    " TFRecord file.")

    parser.add_argument('--annotation_xml',
                        help='The path to the xml annotation file.',
                        default='data/ImageCLEF/training/FigureSeparationTraining2016GT.xml',
                        type=str)

    parser.add_argument('--image_directory_path',
                        help='The path to the directory whre the images are stored.',
                        default='data/ImageCLEF/training/FigureSeparationTraining2016/',
                        type=str)

    parser.add_argument('--output_tfrecord',
                        help='The path of the tfrecord file to which annotations"\
                                " have to be exported.',
                        default='data/ImageCLEF/training/training.tfrecord',
                        type=str)


    return parser.parse_args(args)


def main(args=None):
    """
    Export the ImageCLEF dataset to a TFRecord file.
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
    export_figures_to_tf_record(
        figure_generator=figure_generator,
        tf_record_filename=args.output_tfrecord)


if __name__ == '__main__':
    main()
