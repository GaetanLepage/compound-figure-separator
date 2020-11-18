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
import os
from argparse import ArgumentParser, Namespace
import logging
from typing import List
import csv

from compfigsep.data.figure_generators import IphotodrawXmlFigureGenerator, add_iphotodraw_args

sys.path.append(".")

def parse_args(args: List[str]) -> Namespace:
    """
    Parse the arguments from the command line.

    Args:
        args (List[str]):   The arguments from the command line call.

    Returns:
        namespace (Namespace):  Populated namespace.
    """
    parser = ArgumentParser(description="Convert annotations from individual iPhotoDraw"\
                                        " xml annotation files. to a CSV annotations file.")

    add_iphotodraw_args(parser=parser)

    parser.add_argument('--output_filename',
                        dest="csv_output_filename",
                        help="Name of the csv export file.",
                        default="captions_eval.csv",
                        type=str)

    return parser.parse_args(args)


def main(args: List[str] = None) -> None:
    """
    Load figures from iPhotoDraw xml annotation files and export them to csv.

    Args:
        args (List[str]):   Arguments from the command line.
    """

    # Parse arguments.
    if args is None:
        args = sys.argv[1:]
    parsed_args = parse_args(args)


    if os.path.isfile(parsed_args.csv_output_filename):
        logging.error("File already exists. Aborting.")
        return

    # Create the figure generator handling xml annotation files.
    figure_generator = IphotodrawXmlFigureGenerator(
        file_list_txt=parsed_args.file_list_txt,
        image_directory_path=parsed_args.image_directory_path)

    csv_row: List[str] = []
    labels_list: List[str] = []

    with open(parsed_args.csv_output_filename, 'w') as csv_output_file:

        csv_writer = csv.writer(csv_output_file, delimiter='\t')

        for figure in figure_generator():

            figure.load_caption_annotation()

            if hasattr(figure, 'gt_subcaptions'):
                labels_list = list(figure.gt_subcaptions.keys())
                print(labels_list)
            else:
                labels_list = []

            csv_row = [
                figure.image_filename,
                figure.caption if hasattr(figure, 'caption') else '',
                ', '.join(labels_list)
            ]

            if hasattr(figure, 'gt_subcaptions'):
                csv_row.extend(figure.gt_subcaptions.values())
                csv_row.extend('' for _ in range(20 - len(figure.gt_subcaptions)))
            else:
                csv_row.extend('' for _ in range(20))

            csv_writer.writerow(csv_row)




if __name__ == '__main__':
    main()
