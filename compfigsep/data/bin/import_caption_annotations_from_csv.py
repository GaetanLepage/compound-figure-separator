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


#####################################
Ingest caption splitting annotations.
"""

import sys
from argparse import ArgumentParser, Namespace

from typing import List, Optional

import csv

from compfigsep.data.figure_generators import FigureGenerator
from compfigsep.utils.figure.label import (LabelStructure,
                                           LabelStructureEnum,
                                           map_label)
from compfigsep.caption_splitting.replaceutf8 import replace_utf8
from compfigsep.data.figure_generators.json_figure_generator import JsonFigureGenerator

sys.path.append(".")

def parse_args(args: List[str]) -> Namespace:
    """
    Parse the arguments from the command line.

    Args:
        args (List[str]):   The arguments from the command line call.

    Returns:
        namespace (Namespace):  Populated namespace.
    """
    parser: ArgumentParser = ArgumentParser(description="TODO")


    parser.add_argument('--initial_json',
                        dest="initial_json",
                        help="Name of the json file of the initial data set.",
                        type=str)

    parser.add_argument('--annotation_csv',
                        dest="annotation_csv",
                        help="Name of the csv annotation file.",
                        type=str)

    parser.add_argument('--output_filename',
                        dest="json_output_filename",
                        help="Name of the json export file.",
                        type=str)


    return parser.parse_args(args)


def ingest_caption_splitting_annotations(figure_generator: FigureGenerator,
                                         annotation_csv: str,
                                         json_output_filename: str) -> None:
    """
    Ingest the manual annotations for caption splitting and add them to an existing data set.

    Args:
        figure_generator (FigureGenerator):     A generator yielding figure objects
        json_output_filename (str):             Name of the JSON output file.
        json_output_directory (str):            Path to the directory where to save the JSON
                                                    output file.
    """
    with open(annotation_csv, 'r') as csv_annotation_file:
        csv_reader = csv.reader(csv_annotation_file, delimiter='\t')

        rows: List[List[str]] = [row for row in csv_reader][1:]


    for figure in figure_generator(random_order=False):

        print('#############')
        print('->', figure.image_filename)

        # Find the corresponding row in the annotation file.
        matched_row: Optional[List[str]] = None

        for row_index, row in enumerate(rows):

            if row[0] == figure.image_filename:
                matched_row = rows.pop(row_index)
                break

        # Assert we found a matching row.
        assert matched_row is not None

        # Get the caption text from the annotation.
        annotation_caption_text: str = matched_row[1].strip()

        # Case where the figure has no captiopn
        if not hasattr(figure, 'caption') or figure.caption == "":
            assert annotation_caption_text == "", annotation_caption_text

            # Nothing to do: go to next figure
            continue

        # Get the caption stored in the Figure object.
        figure.caption = figure.caption.strip()

        # Check if both caption texts match
        if annotation_caption_text != figure.caption:
            print(f"{annotation_caption_text}\n!=\n{figure.caption}\n{len(annotation_caption_text)} " \
                  f"|| {len(figure.caption)}")

        label_text_list: List[str] = [label.strip() for label in matched_row[2].split(',')]

        gt_label_list: List[str] = [map_label(subfigure.label.text)
                                    for subfigure in figure.gt_subfigures
                                    if hasattr(subfigure, 'caption')
                                        and subfigure.label is not None]


        gt_labels_structure = LabelStructure.from_labels_list(gt_label_list)
        gt_labels_structure.num_labels = len(figure.gt_subfigures)
        # print("gt label structure:", gt_labels_structure)
        # print("raw GT label list: ", gt_label_list)
        gt_label_list = gt_labels_structure.get_core_label_list()

        assert gt_label_list == gt_labels_structure.get_core_label_list(), gt_labels_structure


        subcaptions: List[str] = [cell_text.strip()
                                  for cell_text in matched_row[3:]
                                  if cell_text != '']


        # Case where there is no labels : ['_', '_', '_']
        if gt_labels_structure.labels_type == LabelStructureEnum.NONE:
            assert len(label_text_list) == len(figure.gt_subfigures)

            print("panel(s) without label(s)")
            print(f"subcaptions={subcaptions}")
            print(f"len={len(subcaptions)}")
            print(f"len_subfigures: {len(figure.gt_subfigures)}")

#             if not subcaptions:
#                 assert len(subcaptions) == 0
#
#                 figure.gt_subfigures[0].caption = figure.caption

            for subfigure in figure.gt_subfigures:
                subfigure.caption = "TODO"

            continue

        # Case 2: explicit labels: ['A', 'B', 'C']

        assert len(subcaptions) == len(figure.gt_subfigures)

        if len(label_text_list) != len(gt_label_list):
            print(figure.image_filename)
            print("No labels")

            print("GT label list: ", gt_label_list)
            print("Annotation label list: ", label_text_list)

            print("PROBLEM")




def main(args: List[str] = None) -> None:
    """
    TODO

    Args:
        args (List[str]):   Arguments from the command line.
    """

    # Parse arguments.
    if args is None:
        args = sys.argv[1:]
    parsed_args: Namespace = parse_args(args)

    figure_generator: FigureGenerator = JsonFigureGenerator(
        json_path=parsed_args.initial_json,
        default_random_order=False)

    ingest_caption_splitting_annotations(figure_generator=figure_generator,
                                         annotation_csv=parsed_args.annotation_csv,
                                         json_output_filename=parsed_args.json_output_filename)


if __name__ == '__main__':
    main()
