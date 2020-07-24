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


###################################################################################
Definition of an abstract class to handle figure data loading from various sources.
"""

import os
from abc import ABC, abstractmethod
from typing import Iterable

import compfigsep

from ...utils.figure.figure import Figure

# Localize data folder
PROJECT_DIR = os.path.join(os.path.dirname(compfigsep.__file__),
                           os.pardir)
DATA_DIR = os.path.join(PROJECT_DIR,
                        "data/")
DATA_DIR = os.path.realpath(DATA_DIR)


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
    def __call__(self) -> Iterable[Figure]:
        """
        Abstract method that has to be implemented.
        It has to be an iterable (generator function) that yields Figure objects.

        Returns:
            Iterable[Figure]:   figure objects with or without annotations (and or detections).
        """
        raise NotImplementedError('This method has to be implemented for each subclass.')
