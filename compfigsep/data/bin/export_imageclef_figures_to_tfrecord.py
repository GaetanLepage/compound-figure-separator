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

sys.path.append(".")

from compfigsep.data.figure_generators import add_image_clef_args, ImageClefXmlFigureGenerator
from compfigsep.data.export import export_figures_to_tf_record


def parse_args(args: list[str]) -> Namespace:
    """
    Parse the arguments from the command line.

    Args:
        args (list[str]):   The arguments from the command line call.

    Returns:
        namespace (Namespace):  Populated namespace.
    """
    parser: ArgumentParser = ArgumentParser(
        description="Convert the ImageCLEF dataset to a TFRecord file."
    )

    add_image_clef_args(parser=parser)

    parser.add_argument(
        '--output_tfrecord',
        help="The path of the tfrecord file to which annotations"
             " have to be exported.",
        default="data/ImageCLEF/training/training.tfrecord",
        type=str
    )

    return parser.parse_args(args)


def main(args: list[str] = None) -> None:
    """
    Export the ImageCLEF dataset to a TFRecord file.

    Args:
        args (list[str]):   The arguments from the command line call.
    """

    # Parse arguments.
    if args is None:
        args = sys.argv[1:]

    parsed_args: Namespace = parse_args(args)

    # Create the figure generator handling xml annotation files
    figure_generator: ImageClefXmlFigureGenerator = ImageClefXmlFigureGenerator(
        xml_annotation_file_path=parsed_args.annotation_xml,
        image_directory_path=parsed_args.image_directory_path
    )

    # Export figures to csv
    export_figures_to_tf_record(
        figure_generator=figure_generator,
        tf_record_filename=parsed_args.output_tfrecord
    )


if __name__ == '__main__':
    main()
