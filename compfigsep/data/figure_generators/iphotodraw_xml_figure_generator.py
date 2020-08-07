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


######################################################
Figure generator handling the PanelSeg (Zou) data set.
"""

from __future__ import annotations
import os
import sys
import logging
from typing import Iterable, List
from argparse import ArgumentParser

import progressbar # type: ignore

from ...utils.figure.figure import Figure
from .figure_generator import FigureGenerator


def add_iphotodraw_args(parser: ArgumentParser,
                        default_eval_list_path: str = None):
    """
    Parse the argument for loading a json file.

    Args:
        parser (ArgumentParser):        An ArgumentParser.
        default_eval_list_path (str):   Default path to a txt file list.
    """

    if default_eval_list_path is None or not os.path.isfile(default_eval_list_path):

        default_eval_list_path = "data/zou/eval.txt"

    parser.add_argument('--file_list_txt',
                        help="The path to the txt file listing the images.",
                        default=default_eval_list_path,
                        type=str)

    parser.add_argument('--image_directory_path',
                        help="The path to the directory where the images are stored.",
                        default=None,
                        type=str)



class IphotodrawXmlFigureGenerator(FigureGenerator):
    """
    Generator of Figure objects from iPhotoDraw xml annotations (PanelSeg data set).
    The input files can be provided either from a csv list or from the path to the directory
    where the image files are stored.

    Attributes:
        data_dir (str):             The path to the directory where the image data sets are
                                        stored.
        image_paths (List[str]):    List of the image paths.
    """

    def __init__(self,
                 file_list_txt: str = None,
                 image_directory_path: str = None,
                 image_paths_list: List[str] = None,
                 caption_annotation_file: str = None) -> None:
        """
        Args:
            file_list_txt (str):            The path of the list of figures which annotations
                                                have to be loaded.
            image_directory_path (str):     The path of the directory where the images are stored
            caption_annotation_file (str):  The path to the caption annotation file.
        """
        # Call base class method.
        super().__init__()

        # If a list of image paths was provided.
        if image_paths_list is not None:
            self.image_paths = image_paths_list

        # If a list of image files was provided, read it and store the image files.
        elif file_list_txt is not None:

            # Read list of image files
            with open(file_list_txt, 'r') as eval_list_file:
                eval_list_lines = eval_list_file.read().splitlines()

            self.image_paths = [line if os.path.isfile(line)
                                else os.path.join('data/', line)
                                for line in eval_list_lines]

        # If a path was provided, list the image files in this directory.
        elif image_directory_path is not None:

            self.image_paths = [f for f in os.listdir(image_directory_path)
                                if f.endswith('.jpg') and os.path.isfile(
                                    os.path.join(image_directory_path, f))]

        else:
            logging.error("Either one of the options has to be set.")
            sys.exit(1)


        # Caption annotations
        if caption_annotation_file is not None:
            with open(caption_annotation_file, 'r') as caption_annotation_file_:
                self.caption_lines = caption_annotation_file_.readlines()


    def __copy__(self) -> IphotodrawXmlFigureGenerator:

        return IphotodrawXmlFigureGenerator(image_paths_list=self.image_paths)


    def __call__(self) -> Iterable[Figure]:
        """
        'Generator' method yielding annotated figures from the PanelSeg data set.

        Returns:
            Iterable[Figure]:   Figure objects with annotations.
        """

        # Looping over the list of image paths.
        for image_index, image_path in enumerate(progressbar.progressbar(self.image_paths)):

            # Create figure object.
            figure = Figure(image_path=image_path,
                            index=image_index)

            # Load image file.
            try:
                figure.load_image()
            except FileNotFoundError as exception:
                logging.error(exception)
                continue

            # Load annotation file.
            xml_path = figure.image_path.replace('.jpg', '_data.xml')
            # Load figure annotations.
            figure.load_annotation_from_iphotodraw(xml_path)

            # Load the caption and, if available, the annotations.
            figure.load_caption_annotation()

            yield figure
