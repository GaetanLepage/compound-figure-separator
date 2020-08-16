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

from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import Iterable, Callable
import copy

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
    """

    def __init__(self) -> None:
        self.data_dir = DATA_DIR


    @abstractmethod
    def __copy__(self) -> FigureGenerator:
        raise NotImplementedError('This method has to be implemented for each subclass.')


    @abstractmethod
    def __call__(self) -> Iterable[Figure]:
        """
        Abstract method that has to be implemented.
        It has to be an iterable (generator function) that yields Figure objects.

        Returns:
            Iterable[Figure]:   figure objects with or without annotations (and or detections).
        """
        raise NotImplementedError('This method has to be implemented for each subclass.')


class StackedFigureGenerator(FigureGenerator):
    """
    Class for creating a FigureGenerator by applying a function to all the figures from an
    existing FigureGenerator.

    Attributes:
        base_figure_generator (FigureGenerator):    A figure generator yielding Figure
                                                        objects.
        function (Callable[[Figure], None]):        A function taking a Figure as argument and
                                                            modifying it.
    """

    def __init__(self,
                 base_figure_generator: FigureGenerator,
                 function: Callable[[Figure], None]) -> None:
        """
        Args:
            base_figure_generator (FigureGenerator):    A figure generator yielding Figure
                                                            objects.
            function (Callable[[Figure], None]):        A function taking a Figure as argument and
                                                            modifying it.
        """
        super().__init__()

        self._base_figure_generator = base_figure_generator
        self._function = function


    def __copy__(self) -> StackedFigureGenerator:

        return StackedFigureGenerator(
            base_figure_generator=copy.copy(self._base_figure_generator),
            function=self._function)


    def __call__(self) -> Iterable[Figure]:

        from pprint import pprint

        for figure in self._base_figure_generator():

            print("-- Stacked figure generator --")

            self._function(figure)

            pprint(figure.to_dict())

            yield figure
