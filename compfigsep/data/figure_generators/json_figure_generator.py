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


#################################################
Figure generator handling a JSON annotation file.
"""

import os
import logging
import json
import progressbar

from ...utils.figure import (
    Figure,
    SubFigure,
    DetectedSubFigure,
    Panel,
    Label)

from .figure_generator import FigureGenerator


class JsonFigureGenerator(FigureGenerator):
    """
    Generator of Figure objects from a JSON annotation file.

    Attributes:
        json_annotation_file_path (str):    The path to the json annotation file.
    """

    def __init__(self, json_annotation_file_path: str):
        """
        Init function.
        Call the init function of the abstract parent class.

        Args:
            json_annotation_file_path (str):    The path to the json annotation file.
        """

        self.json_annotation_file_path = json_annotation_file_path

        super().__init__()

        if not os.path.isfile(json_annotation_file_path):
            raise FileNotFoundError("The annotation json file does not exist :"\
                "\n\t {}".format(json_annotation_file_path))


    def __call__(self) -> Figure:
        """
        Generator of Figure objects from a json annotation file.

        Yields:
            figure (Figure):    Figure objects with annotations.
        """


        with open(self.json_annotation_file_path, 'r') as json_annotation_file:
            data_dict = json.load(json_annotation_file)

        for index, figure_dict in enumerate(progressbar.progressbar(data_dict.values())):


            yield Figure.from_dict(figure_dict=figure_dict,
                                   index=index)
