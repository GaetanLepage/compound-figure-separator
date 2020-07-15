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

Collaborators:  Niccolò Marini (niccolo.marini@hevs.ch)
                Stefano Marchesin (stefano.marchesin@unipd.it)


##################################################################################
Script to perform and evaluate the panel splitting task on the ImageCLEF data set.

TODO remove this file
"""

import sys
from argparse import ArgumentParser
from typing import List

sys.path.append(".")

from compfigsep.data.figure_generators import ImageClefXmlFigureGenerator
from compfigsep.panel_splitting.predict import predict
from compfigsep.panel_splitting.evaluate import evaluate_detections


def parse_args(args: List[str]) -> ArgumentParser:
    """
    Parse the arguments from the command line.

    Args:
        args (List[str]):   The arguments from the command line call.

    Returns:
        parser (ArgumentParser):   Populated namespace.
    """
    parser = ArgumentParser(description="Run panel splitting predictions for the"\
                                        " ImageCLEF dataset and evaluate performance.")

    parser.add_argument('--annotation_xml',
                        help="The path to the xml annotation file.",
                        default="data/ImageCLEF/test/FigureSeparationTest2016GT.xml",
                        type=str)

    parser.add_argument('--image_directory_path',
                        help="The path to the directory where the images are stored.",
                        default="data/ImageCLEF/test/FigureSeparationTest2016/",
                        type=str)

    return parser.parse_args(args)


def main(args: List[str] = None):
    """
    Load figures from ImageCLEF xml annotation files and test a model for the
    panel splitting task.

    Args:
        args (List[str]):   The arguments from the command line call.
    """

    # Parse arguments.
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Create the figure generator handling xml annotation files
    figure_generator = ImageClefXmlFigureGenerator(
        xml_annotation_file_path=args.annotation_xml,
        image_directory_path=args.image_directory_path)

    # Augment Figures by predicting panel locations
    augmented_figure_generators = predict(figure_generator=figure_generator,
                                          predict_function=None,
                                          pre_processing_function=None)

    # Evaluate predictions
    evaluate_detections(figure_generator=augmented_figure_generators)


if __name__ == '__main__':
    main()
