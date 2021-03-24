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


##################################
Script to perform panel splitting.
"""

import sys
import os
import copy
import logging
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

from typing import List, Dict
from compfigsep.utils.figure import Figure, DetectedPanel


from compfigsep.data.figure_generators import (JsonFigureGenerator,
                                               add_json_arg,
                                               StackedFigureGenerator)

from compfigsep.panel_splitting import (evaluate_detections,
                                        panel_filtering)

from compfigsep.data.export import export_figures_to_json

import compfigsep
sys.path.append('.')

MODULE_DIR = os.path.dirname(compfigsep.__file__)


def _parse_args(args: List[str]) -> Namespace:
    """
    Parse the arguments from the command line.

    Args:
        args (List[str]):   The arguments from the command line call.

    Returns:
        parser (Namespace): Populated namespace.
    """
    parser: ArgumentParser = ArgumentParser(
        description="Evaluate panel splitting.")

    add_json_arg(parser=parser,
                 json_default_relative_path=\
                    '../data/zou/eval.json')

    return parser.parse_args(args)


def predict_panels(figure: Figure) -> None:
   """
   Apply the full caption splitting pipeline to the given figure.
   The subcaptions detections are stored in the `detected_subcaptions` attribute.

   Args:
       figure (Figure):    A figure object.
   """

   filtered_panels: List[DetectedPanel] = panel_filtering.filter_panels(
       panel_list=figure.detected_panels)

   figure.detected_panels= filtered_panels



def main(args: List[str] = None) -> None:
    """
    Launch detection and evaluation of the panel splitting task on a JSON data set.

    Args:
        args (List[str]):   Arguments from the command line.
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
        function=predict_panels)

    logging.info("Exporting detected captions")

    # Export detections to JSON.
    export_figures_to_json(figure_generator=copy.copy(figure_generator),
                           json_output_directory="compfigsep/caption_splitting/output/")

    logging.info("Evaluate detections")

    # Evaluate the data set.
    evaluate_detections(figure_generator=prediction_figure_generator)


if __name__ == '__main__':
    main()
