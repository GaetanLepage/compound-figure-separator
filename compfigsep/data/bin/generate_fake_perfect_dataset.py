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
Script to generate a dataset emulating perfect detection.
The ground truth information are injected in the `detected*` fields of the figure objects.
"""

import sys
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

from compfigsep.data.figure_generators.json_figure_generator import (
    add_json_arg,
    JsonFigureGenerator
)
from compfigsep.data.figure_generators.figure_generator import (
    FigureGenerator,
    StackedFigureGenerator
)
from compfigsep.data.export import export_figures_to_json
from compfigsep.utils.figure.sub_figure import DetectedSubFigure


def parse_args(args: list[str]) -> Namespace:
    """
    Parse the arguments from the command line.

    Args:
        args (list[str]):   The arguments from the command line call.

    Returns:
        parser (Namespace): Populated namespace.
    """
    parser: ArgumentParser = ArgumentParser(
        description="Evaluate panel splitting detections."
    )

    add_json_arg(
        parser=parser,
        json_default_path='data/zou/eval.json'
    )

    parser.add_argument(
        '--output-filename',
        dest="json_output_filename",
        help="Name of the json export file.",
        default="fake_perfect_detections.json",
        type=str
    )

    return parser.parse_args(args)


def _edit_figure(figure) -> None:
    figure.detected_panels = []
    figure.detected_labels = []
    figure.detected_subcaptions = OrderedDict()
    figure.detected_subfigures = []

    for gt_subfigure in figure.gt_subfigures:
        detected_subfigure: DetectedSubFigure = DetectedSubFigure.from_normal_sub_figure(
            subfigure=gt_subfigure
        )

        figure.detected_subfigures.append(detected_subfigure)

        assert detected_subfigure.panel is not None
        figure.detected_panels.append(detected_subfigure.panel)

        if detected_subfigure.label is not None:
            figure.detected_labels.append(detected_subfigure.label)

        if detected_subfigure.caption is not None:
            label_text: str = '_'
            if detected_subfigure.label is not None and detected_subfigure.label.text != '':
                label_text = detected_subfigure.label.text
            figure.detected_subcaptions[label_text] = detected_subfigure.caption

        # Subcaptions
        if hasattr(figure, 'gt_subcaptions'):
            figure.detected_subcaptions = figure.gt_subcaptions.copy()


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

    # Create the figure generator from the source data.
    input_figure_generator: JsonFigureGenerator = JsonFigureGenerator(json_path=parsed_args.json)

    output_figure_generator: FigureGenerator = StackedFigureGenerator(
        base_figure_generator=input_figure_generator,
        function=_edit_figure
    )

    # Export figures back to json.
    export_figures_to_json(
        figure_generator=output_figure_generator,
        json_output_filename=parsed_args.json_output_filename
    )


if __name__ == '__main__':
    main()
