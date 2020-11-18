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


############################################################################
Script to export the ImageCLEF dataset to a tfrecord file compatible with
Tensorflow's Object Detection API.
(https://github.com/tensorflow/models/tree/master/official/vision/detection)
"""

import sys
from argparse import ArgumentParser, Namespace

from typing import List

sys.path.append(".")

from compfigsep.data.figure_generators import ImageClefXmlFigureGenerator
from compfigsep.data.export import export_figures_to_tf_record


def parse_args(args: List[str]) -> Namespace:
    """
    Parse the arguments from the command line.

    Args:
        args (List[str]):   The arguments from the command line call.

    Returns:
        namespace (Namespace):  Populated namespace.
    """
    parser = ArgumentParser(description="Convert the ImageCLEF dataset to a"\
                                        " TFRecord file.")

    parser.add_argument('--annotation_xml',
                        help="The path to the xml annotation file.",
                        default="data/ImageCLEF/training/FigureSeparationTraining2016GT.xml",
                        type=str)

    parser.add_argument('--image_directory_path',
                        help="The path to the directory where the images are stored.",
                        default="data/ImageCLEF/training/FigureSeparationTraining2016/",
                        type=str)

    parser.add_argument('--output_tfrecord',
                        help="The path of the tfrecord file to which annotations"\
                             " have to be exported.",
                        default="data/ImageCLEF/training/training.tfrecord",
                        type=str)


    return parser.parse_args(args)


def main(args: List[str] = None):
    """
    Export the ImageCLEF dataset to a TFRecord file.

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

    # Export figures to csv
    export_figures_to_tf_record(figure_generator=figure_generator,
                                tf_record_filename=args.output_tfrecord)


if __name__ == '__main__':
    main()
