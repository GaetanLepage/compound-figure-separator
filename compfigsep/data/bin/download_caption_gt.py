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

Collaborator:   Niccolò Marini (niccolo.marini@hevs.ch)


#############################################################
Script to Download caption data from PubMedCentral data base.
"""

import sys
import os
import logging
import urllib.request
from typing import List
from argparse import ArgumentParser
import progressbar

import xml.etree.ElementTree as ET

from compfigsep.data.figure_generators import DATA_DIR


# API url to download xml
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&retmode=xml&id="


def parse_args(args: List[str]) -> ArgumentParser:
    """
    Parse the arguments from the command line.

    Args:
        args (List[str]):   The arguments from the command line call.

    Returns:
        parser (ArgumentParser):   Populated namespace.
    """
    parser = ArgumentParser(description="Convert the ImageCLEF dataset to a"\
                                        " TFRecord file.")

    parser.add_argument('--file_list_txt',
                        help="The path to the txt file listing the images.",
                        default="data/zou/eval.txt",
                        type=str)

    # TODO
    parser.add_argument('--override',
                        help="Override caption txt file if exists.",
                        action='store_true')

    return parser.parse_args(args)


def get_captions(file_list_txt: str,
                 override: bool = False):
    """
    TODO

    Args:
        file_list_txt (str):    TODO.
        override (bool):        TODO.
    """

    # Read the file list and store the image names.
    if file_list_txt is not None:

        # Read list of image files
        with open(file_list_txt, 'r') as list_file:
            image_paths = [os.path.join(DATA_DIR, path)
                           for path in list_file.read().splitlines()]

    pmc_id = ''
    for image_path in progressbar.progressbar(image_paths):

        # Get image filename from path (and remove the extension).
        image_filename = os.path.splitext(os.path.basename(image_path))[0]

        # This script only manages images from PubMedCentral.
        if image_filename.startswith('PMC'):

            # Extract PMC id and original image file name from the full file name.
            # The file name format is : PMCxxxxx_yyyyyy.jpg
            # new_pmc_id = PMCxxxxx
            # target_file_name = yyyyyy
            new_pmc_id, target_file_name = image_filename.split('_', maxsplit=1)


            # Path to the txt file that will contain the caption text.
            caption_annotation_file_path = image_path.replace('.jpg', '_caption.txt')

            if os.path.isfile(caption_annotation_file_path) and not override:
                logging.warning(f"Caption file {caption_annotation_file_path} for image"\
                                f" {image_path} already exists."\
                                "\n==> As `override` has been set to False,"\
                                " this image is skipped.")
                continue


            # If it is the same article as the previous iteration, no need to re-download the
            # xml data.
            if new_pmc_id != pmc_id:
                pmc_id = new_pmc_id
                xml_url = BASE_URL + pmc_id
                xml_data = urllib.request.urlopen(xml_url).read()
                xml_root = ET.fromstring(xml_data)

            for figure_element in xml_root.iter('fig'):
                graphic_element = figure_element.find('graphic')
                # If image name is not available.
                if graphic_element is None:
                    continue
                file_base_name = next(v for k, v in graphic_element.attrib.items()
                                      if k.endswith('href'))

                # No need to go further if this figure_element corresponds to another figure.
                if file_base_name != target_file_name:
                    found_matching_figure_element = False
                    continue

                found_matching_figure_element = True

                caption_element = figure_element.find('caption')
                if caption_element is not None:

                    # Gather all the text (possibly nested) from the caption_element.
                    caption_text = ''.join(caption_element.itertext()).lstrip()

                    # Remove multiple spaces and '\t' from the caption.
                    caption_text = ' '.join(caption_text.split())

                    with open(caption_annotation_file_path, 'w') as output_file:
                        output_file.write(caption_text)

                # Arriving at this line means the caption has been succesfully written.
                # No need to keep looping over the figures from this article.
                break

            if not found_matching_figure_element:
                logging.warning(f"No match for image file name {target_file_name}."\
                                f"\nPMC id = {pmc_id}")

        else:
            logging.warning(f"Not a PMC article: {image_filename}")


def main(args: List[str] = None):
    """
    TODO

    Args:
        args (List[str]):   Arguments from the command line.
    """

    # Parse arguments.
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Get the captions
    get_captions(file_list_txt=args.file_list_txt,
                 override=args.override)

if __name__ == '__main__':
    main()
