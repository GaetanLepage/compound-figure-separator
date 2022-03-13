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


#############################################
Script to perform compound figure separation.
"""

import sys
import os
import copy
import logging
from argparse import ArgumentParser, Namespace

from compfigsep.data.figure_generators import (JsonFigureGenerator,
                                               add_json_arg,
                                               StackedFigureGenerator)

from compfigsep.caption_splitting import (label_identification,
                                          label_expansion,
                                          label_filtering,
                                          extract_subcaptions,
                                          evaluate_detections)

from compfigsep.data.export import export_figures_to_json

import compfigsep
sys.path.append('.')

MODULE_DIR = os.path.dirname(compfigsep.__file__)


def _parse_args(args: list[str]) -> Namespace:
    """
    Parse the arguments from the command line.

    Args:
        args (list[str]):   The arguments from the command line call.

    Returns:
        parser (Namespace): Populated namespace.
    """
    parser: ArgumentParser = ArgumentParser(
        description="Evaluate caption splitting on the prostate data set.")

    add_json_arg(parser=parser,
                 json_default_relative_path='../data/pubmed_caption_splitting/'
                                            'prostate_data_only_annotated_captions.json')

    return parser.parse_args(args)


def main(args: list[str] = None) -> None:
    """
    Launch detection and evaluation of the compound figure separation task on a JSON data set.

    Args:
        args (list[str]):   Arguments from the command line.
    """

    # Parse arguments.
    if args is None:
        args = sys.argv[1:]
    parsed_args: Namespace = _parse_args(args)

    # Create the figure generator handling JSON annotation files.
    figure_generator: JsonFigureGenerator = JsonFigureGenerator(
        json_path=parsed_args.json)

    prediction_figure_generator: StackedFigureGenerator = StackedFigureGenerator(
        base_figure_generator=figure_generator,
        function=predict_caption)

    logging.info("Exporting detected captions")

    # Export detections to JSON.
    export_figures_to_json(figure_generator=copy.copy(prediction_figure_generator),
                           json_output_directory="compfigsep/caption_splitting/output/")

    logging.info("Evaluate detections")

    # Evaluate the data set.
    evaluate_detections(figure_generator=prediction_figure_generator)


if __name__ == '__main__':
    main()
