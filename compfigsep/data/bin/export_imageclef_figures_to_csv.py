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


###########################################################################
Script to export the ImageCLEF dataset annotations to a csv compatible with
keras-retinanet (https://github.com/fizyr/keras-retinanet).
"""

import sys
from argparse import ArgumentParser, Namespace

from compfigsep.data.figure_generators import add_image_clef_args, ImageClefXmlFigureGenerator
from compfigsep.data.export import export_figures_to_csv

sys.path.append(".")


def parse_args(args: list[str]) -> Namespace:
    """
    Parse the arguments from the command line.

    Args:
        args (list[str]): The arguments from the command line call.

    Returns:
        namespace (Namespace):  Populated namespace.
    """
    parser: ArgumentParser = ArgumentParser(
        description="Convert annotations from an ImageCLEF"
                    " xml annotation file to a CSV annotations file."
    )

    add_image_clef_args(parser=parser)

    parser.add_argument(
        '--output_csv',
        help="The path of the csv file to which annotations have to be exported.",
        default="data/imageCLEF/test/test.csv",
        type=str
    )

    parser.add_argument(
        '--individual_csv',
        help="Also export the annotations to a single csv file.",
        action='store_true'
    )

    parser.add_argument(
        '--individual_export_csv_directory',
        help="The path of the directory where to store the individual csv annotation files.",
        default="data/imageCLEF/test/test.csv",
        type=str
    )

    return parser.parse_args(args)


def main(args: list[str] = None) -> None:
    """
    Load figures from ImageCLEF xml annotation files and export them to csv.

    Args:
        args (list[str]):   Arguments from the command line.
    """

    # Parse arguments.
    if args is None:
        args = sys.argv[1:]
    parsed_args: Namespace = parse_args(args)

    # Create the figure generator handling xml annotation files
    figure_generator = ImageClefXmlFigureGenerator(
        xml_annotation_file_path=parsed_args.annotation_xml,
        image_directory_path=parsed_args.image_directory_path
    )

    # Export figures to csv
    export_figures_to_csv(
        figure_generator=figure_generator,
        output_csv_file=parsed_args.output_csv,
        individual_export=parsed_args.individual_csv,
        individual_export_csv_directory=parsed_args.individual_export_csv_directory
    )


if __name__ == '__main__':
    main()
