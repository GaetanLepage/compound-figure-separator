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


###########################################
Figure generator handling a list of images.
"""

from __future__ import annotations
import os
import logging
import random

from typing import Iterable, Optional, List

from ...utils.figure.figure import Figure
from .figure_generator import FigureGenerator


class ImageListFigureGenerator(FigureGenerator):
    """
    Generator of Figure objects from an image list.
    This generator does not load any annotations.

    Attributes:
        image_list_txt (str):           The path to the list of images to be loaded.
        image_directory_path (str):     The path to the directory where the images are stored.
        default_random_order (bool):    Wether to yield figures in a random order.
    """

    def __init__(self,
                 image_list_txt: str,
                 image_directory_path: str = None,
                 default_random_order: bool = False) -> None:
        """
        Args:
            image_list_txt (str):           The path to the list of images to be loaded.
            image_directory_path (str):     The path to the directory where the images are stored.
            default_random_order (bool):    Wether to yield figures in a random order.
        """
        super().__init__(default_random_order=default_random_order)

        if not os.path.isfile(image_list_txt):
            raise FileNotFoundError("The evaluation list file does not exist :"\
                                    "\n\t {}".format(image_list_txt))

        self.image_directory_path: Optional[str] = image_directory_path

        self.image_list_txt: str = image_list_txt


    def __copy__(self) -> ImageListFigureGenerator:
        return ImageListFigureGenerator(image_list_txt=self.image_list_txt,
                                        image_directory_path=self.image_directory_path,
                                        default_random_order=self.default_random_order)


    def __call__(self, random_order: Optional[bool] = None) -> Iterable[Figure]:

        with open(self.image_list_txt, 'r') as image_list_file:

            lines: List[str] = image_list_file.readlines()

        if random_order is None:
            random_order = self.default_random_order

        if random_order:
            random.shuffle(lines)

        for image_counter, line in enumerate(lines):

            # Compute image path.
            if self.image_directory_path is not None:
                image_file_path: str = os.path.join(self.image_directory_path, line[:-1])
            elif os.path.isfile(line):
                image_file_path = line
            else:
                image_file_path = os.path.join('data/', line)

            if not os.path.isfile(image_file_path):
                logging.warning("File not found : %s", image_file_path)
                continue

            figure: Figure = Figure(image_path=image_file_path,
                                    index=image_counter)

            figure.load_image()

            yield figure
