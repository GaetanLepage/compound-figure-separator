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


#######################################################
Figure generator handling the pubmed prostate data set.
This set contains gt annotations for captions
"""

import os
import logging

from ...utils.figure.figure import Figure
from .figure_generator import ImageListFigureGenerator


class ProstateCaptionFigureGenerator(ImageListFigureGenerator):
    """
    Generator of Figure objects from the PubMed prostate data set.
    This generator loads ground truth caption annotations.

    Attributes:
        image_list_txt (str):       The path to the list of images to be loaded.
        image_directory_path (str): The path to the directory where the images are stored.
    """

    def __init__(self,
                 image_list_txt: str,
                 image_directory_path: str = None):
        """
        Args:
            image_list_txt (str):       The path to the list of images to be loaded.
            image_directory_path (str): The path to the directory where the images are stored.
        """
        super().__init__(image_list_txt=image_list_txt,
                         image_directory_path=image_directory_path)


    def __call__(self) -> Figure:

        with open(self.image_list_txt, 'r') as image_list_file:

            for image_counter, line in enumerate(image_list_file.readlines()):

                # Compute image path.
                if self.image_directory_path is not None:
                    image_file_path = os.path.join(self.image_directory_path, line[:-1])
                elif os.path.isfile(line):
                    image_file_path = line
                else:
                    image_file_path = os.path.join('data/', line)

                if not os.path.isfile(image_file_path):
                    logging.warning("File not found : %s", image_file_path)
                    continue

                figure = Figure(image_path=image_file_path,
                                index=image_counter)

                figure.load_image()

                yield figure
