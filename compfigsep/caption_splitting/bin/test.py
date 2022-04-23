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


####################################
Script to perform caption splitting.
"""

import sys
import os
import copy
import logging
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

from compfigsep.utils.figure import Figure, LabelStructure


from compfigsep.data.figure_generators import (JsonFigureGenerator,
                                               add_json_arg,
                                               StackedFigureGenerator)

from compfigsep.caption_splitting import (label_identification,
                                          label_expansion,
                                          extract_subcaptions,
                                          evaluate_detections)

from compfigsep.data.export import export_figures_to_json

import compfigsep

sys.path.append('.')

MODULE_DIR: str = os.path.dirname(compfigsep.__file__)


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


def predict_caption(figure: Figure) -> None:
    """
    Apply the full caption splitting pipeline to the given figure.
    The subcaptions detections are stored in the `detected_subcaptions` attribute.

    Args:
        figure (Figure):    A figure object.
    """

    caption: str = figure.caption

    if caption is None:
        return

    label_dict: dict = label_identification(caption=caption)

    labels_list: list[str] = label_expansion(label_dict)

    label_structure: LabelStructure = LabelStructure.from_labels_list(labels_list=labels_list)

    sub_captions_dict: dict[str, str] = extract_subcaptions(caption=caption,
                                                            label_structure=label_structure)

    figure.detected_subcaptions = OrderedDict(sub_captions_dict)

    # TODO label 'B' is not detected in this example.

    debug_pattern: str = ''
    debug_pattern = "Microanatomy of P. verrucosa. A) 3D-reconstruction of the central nervous "
    debug_pattern = "nTE-2-PyP protects prostatic and penile"
    # Case where no label is detected
    debug_pattern = "Immunohistochemical staining for PTEN, SFPQ and HDAC1 in PCa and BENIGN "
    debug_pattern = "Transition from PIN to invasive carcinoma is seen in ARR2PBCreER"
    debug_pattern = "Histopathological diagnose was acinar type adenocarcinoma"
    debug_pattern = "PPAR-gamma protein expression in benign prostate tissues by "
    if debug_pattern in caption:
        sys.exit()


def main(args: list[str] = None) -> None:
    """
    Launch detection and evaluation of the label recognition task on a JSON data set.

    Args:
        args (list[str]):   Arguments from the command line.
    """

    # Parse arguments.
    if args is None:
        args = sys.argv[1:]
    parsed_args: Namespace = _parse_args(args)

    # Create the figure generator handling JSON annotation files.
    figure_generator: JsonFigureGenerator = JsonFigureGenerator(
        json_path=parsed_args.json
    )

    prediction_figure_generator: StackedFigureGenerator = StackedFigureGenerator(
        base_figure_generator=figure_generator,
        function=predict_caption
    )

    logging.info("Exporting detected captions")

    # Export detections to JSON.
    export_figures_to_json(
        figure_generator=copy.copy(prediction_figure_generator),
        json_output_directory="compfigsep/caption_splitting/output/"
    )

    logging.info("Evaluate detections")

    # Evaluate the data set.
    evaluate_detections(figure_generator=prediction_figure_generator)


if __name__ == '__main__':
    main()
