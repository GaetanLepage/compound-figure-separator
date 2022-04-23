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


###################################################################
Script to export Zou's dataset annotations to a csv compatible with
keras-retinanet (https://github.com/fizyr/keras-retinanet).
"""

import sys
from argparse import ArgumentParser, Namespace

from compfigsep.data.figure_generators import IphotodrawXmlFigureGenerator, add_iphotodraw_args
from compfigsep.data.export import export_figures_to_json

sys.path.append(".")


def parse_args(args: list[str]) -> Namespace:
    """
    Parse the arguments from the command line.

    Args:
        args (list[str]):   The arguments from the command line call.

    Returns:
        namespace (Namespace):  Populated namespace.
    """
    parser: ArgumentParser = ArgumentParser(
        description="Convert annotations from individual iPhotoDraw xml annotation files to a CSV"
                    " annotations file."
    )

    add_iphotodraw_args(parser=parser)

    parser.add_argument(
        '--output_filename',
        dest="json_output_filename",
        help="Name of the json export file.",
        default="eval.json",
        type=str
    )

    return parser.parse_args(args)


def main(args: list[str] = None) -> None:
    """
    Load figures from iPhotoDraw xml annotation files and export them to csv.

    Args:
        args (list[str]):   Arguments from the command line.
    """

    # Parse arguments.
    if args is None:
        args = sys.argv[1:]
    parsed_args: Namespace = parse_args(args)

    # Create the figure generator handling xml annotation files.
    figure_generator: IphotodrawXmlFigureGenerator = IphotodrawXmlFigureGenerator(
        file_list_txt=parsed_args.file_list_txt,
        image_directory_path=parsed_args.image_directory_path
    )

    # Export figures to csv.
    export_figures_to_json(
        figure_generator=figure_generator,
        json_output_filename=parsed_args.json_output_filename,
        json_output_directory="data/zou/"
    )


if __name__ == '__main__':
    main()
