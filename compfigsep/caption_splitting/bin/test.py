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


####################################
Script to perform caption splitting.
"""

import sys
import os
import copy
import logging
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

from typing import List
from compfigsep.utils.figure import Figure, LabelStructure


from compfigsep.data.figure_generators import (JsonFigureGenerator,
                                               add_json_arg,
                                               StackedFigureGenerator)

from compfigsep.caption_splitting import (label_identification,
                                          label_expansion,
                                          label_filtering,
                                          extract_subcaptions,
                                          evaluate_detections)

from compfigsep.data.export import export_figures_to_json

# TODO remove (it was just for testing)
from pprint import pprint

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
    parser = ArgumentParser(description="Evaluate caption splitting on the prostate data set.")

    add_json_arg(parser=parser,
                 json_default_relative_path=\
                    '../data/pubmed_prostate/prostate_data_only_annotated_captions.json')

    return parser.parse_args(args)


def main(args: List[str] = None):
    """
    Launch detection and evaluation of the label recognition task on a JSON data set.

    Args:
        args (List[str]):   Arguments from the command line.
    """

    # Parse arguments.
    if args is None:
        args = sys.argv[1:]
    parsed_args = _parse_args(args)

    # Create the figure generator handling JSON annotation files.
    figure_generator = JsonFigureGenerator(
        json_path=parsed_args.json)

    def predict_caption(figure: Figure) -> None:
        """
        Apply the full caption splitting pipeline to the given figure.
        The subcaptions detections are stored in the `detected_subcaptions` attribute.

        Args:
            figure (Figure):    A figure object.
        """

        print("#############")

        caption = figure.caption

        if caption is None:
            return

        print("Caption:", caption)

        label_dict = label_identification(caption=caption)

        pprint(label_dict)

        labels_list: List[str] = label_expansion(label_dict)

        label_structure: LabelStructure = label_filtering.label_filtering(text_labels=labels_list)

        # TODO: just for testing: remove
        # filtered_labels = labels_alphanum
        # target_regex = RE_CHARACTERS
        # target_regex_pos = RE_CHARACTERS_POS

        sub_captions_dict = extract_subcaptions(caption=caption,
                                                label_structure=label_structure)

        print("# DETECTION #")
        pprint(sub_captions_dict)

        figure.detected_subcaptions = OrderedDict(sub_captions_dict)

        # TODO label 'B' is not detected in this example.

        # if "Microanatomy of P. verrucosa. A) 3D-reconstruction of the central nervous system" in caption:
        # if "nTE-2-PyP protects prostatic and penile" in caption:
        # if "" in caption:
        # if "" in caption:
        # Case where no label is detected
        # if "Immunohistochemical staining for PTEN, SFPQ and HDAC1 in PCa and BENIGN tissues" in caption:
        # if "Transition from PIN to invasive carcinoma is seen in ARR2PBCreER" in caption:
        # if "Histopathological diagnose was acinar type adenocarcinoma" in caption:
        # if "PPAR-gamma protein expression in benign prostate tissues by immunohistochemistry" in caption:
        # if caption.startswith("nTE-2-PyP protects prostatic and penile"):
            # sys.exit()

    prediction_figure_generator = StackedFigureGenerator(
        base_figure_generator=figure_generator,
        function=predict_caption)

    logging.info("Exporting detected captions")

    # Export detections to JSON.
    export_figures_to_json(figure_generator=copy.copy(prediction_figure_generator),
                           json_output_directory="compfigsep/caption_splitting/output/")

    logging.info("Evaluate detections")

    # Evaluate the data set.
    # evaluate_detections(figure_generator=prediction_figure_generator)


if __name__ == '__main__':
    main()
