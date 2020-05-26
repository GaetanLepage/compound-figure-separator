#!/usr/bin/env python3
"""
Script to perform and evaluate the panel splitting task on the ImageCLEF data set.
"""

import sys
import argparse

sys.path.append(".")

from panel_seg.io.figure_generators import image_clef_xml_figure_generator
from panel_seg.panel_split.predict import predict
from panel_seg.panel_split.evaluate import evaluate_predictions


def parse_args(args):
    """
    Parse the arguments from the command line.

    Args:
        args: The arguments from the command line call.

    Returns:
        Populated namespace
    """
    parser = argparse.ArgumentParser(description='Run panel splitting predictions for the"\
                                                " ImageCLEF dataset and evaluate performance.')

    parser.add_argument('--annotation_xml',
                        help='The path to the xml annotation file.',
                        default='data/ImageCLEF/test/FigureSeparationTest2016GT.xml',
                        type=str)

    parser.add_argument('--image_directory_path',
                        help='The path to the directory whre the images are stored.',
                        default='data/ImageCLEF/test/FigureSeparationTest2016/',
                        type=str)

    return parser.parse_args(args)


def main(args=None):
    """
    Load figures from ImageCLEF xml annotation files and test a model for the
    panel splitting task.
    """

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Create the figure generator handling xml annotation files
    figure_generator = image_clef_xml_figure_generator(
        xml_annotation_file_path=args.annotation_xml,
        image_directory_path=args.image_directory_path)

    # Augment Figures by predicting panel locations
    augmented_figure_generators = predict(figure_generator=figure_generator,
                                          predict_function=None,
                                          pre_processing_function=None)

    # Evaluate predictions
    evaluate_predictions(figure_generator=augmented_figure_generators)


if __name__ == '__main__':
    main()
