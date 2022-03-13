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


################################################################################################
Script to visualize the ImageCLEF data set by displaying the images along with the corresponding
bounding boxes.
"""

import sys
import os
from argparse import ArgumentParser, Namespace

from compfigsep.data.figure_generators import add_image_clef_args, ImageClefXmlFigureGenerator
from compfigsep.data.figure_viewer import add_viewer_args, view_data_set

sys.path.append('.')


def parse_args(args: list[str]) -> Namespace:
    """
    Parse the arguments from the command line.

    Args:
        args (list[str]):   The arguments from the command line call.

    Returns:
        namespace (Namespace):  Populated namespace.
    """
    parser: ArgumentParser = ArgumentParser(
        description="Preview all the figures from an ImageCLEF data set."
    )

    add_image_clef_args(parser=parser)

    add_viewer_args(parser=parser)

    return parser.parse_args(args)


def main(args: list[str] = None) -> None:
    """
    Launch previsualization of ImageCLEF data set.

    Args:
        args (list[str]):   Arguments from the command line.
    """
    # Parse arguments.
    if args is None:
        args = sys.argv[1:]
    parsed_args: Namespace = parse_args(args)

    # Create the figure generator handling ImageCLEF xml annotation files.
    figure_generator: ImageClefXmlFigureGenerator = ImageClefXmlFigureGenerator(
        xml_annotation_file_path=parsed_args.annotation_xml,
        image_directory_path=parsed_args.image_directory_path,
        default_random_order=parsed_args.random_orders
    )

    # Preview the data set.
    view_data_set(
        figure_generator=figure_generator,
        mode=parsed_args.mode,
        save_preview=parsed_args.save_preview,
        preview_folder=os.path.join(parsed_args.image_directory_path, "preview"),
        delay=parsed_args.delay,
        window_name="ImageCLEF data preview"
    )


if __name__ == '__main__':
    main()
