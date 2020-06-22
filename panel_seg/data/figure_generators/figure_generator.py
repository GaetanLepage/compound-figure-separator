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


###################################################################################
Definition of an abstract class to handle figure data loading from various sources.
"""

import os
from abc import ABC, abstractmethod

import panel_seg

from ...utils.figure.figure import Figure

# Localize data folder
PROJECT_DIR = os.path.join(
    os.path.dirname(panel_seg.__file__),
    os.pardir)
DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data/")


class FigureGenerator(ABC):
    """
    Abstract class representing a figure generator.
    A FigureGenerator is a callable yielding Figure objects.

    Attributes:
        data_dir (str):         The path to the directory where the image data sets are stored.
        current_index (int):    Index of the currently handled figure. This helps knowing the
                                     "progression" of the data loading process.
    """

    def __init__(self):
        """
        Init function for every FigureGenerator.
        """

        self.data_dir = DATA_DIR
        self.current_index = 0


    @abstractmethod
    def __call__(self) -> Figure:
        """
        Abstract method that has to be implemented.
        It has to be an iterable (generator function) that yields Figure objects.

        Yields:
            Figure: figure objects with or without annotations (and or detections).
        """
        raise NotImplementedError('This method has to be implemented for each subclass.')
