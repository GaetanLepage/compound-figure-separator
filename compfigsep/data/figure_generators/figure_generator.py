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


###################################################################################
Definition of an abstract class to handle figure data loading from various sources.
"""

from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import Iterable, Callable, Optional
import copy
from argparse import ArgumentParser

import compfigsep

from ...utils.figure.figure import Figure

# Localize data folder
PROJECT_DIR: str = os.path.join(os.path.dirname(compfigsep.__file__),
                                os.pardir)
DATA_DIR: str = os.path.join(PROJECT_DIR, "data/")
DATA_DIR = os.path.realpath(DATA_DIR)


def add_common_figure_generator_args(parser: ArgumentParser) -> None:
    """
    Parse the argument for loading a json file.

    Args:
        parser (ArgumentParser):        An ArgumentParser.
        default_eval_list_path (str):   Default path to a txt file list.
    """
    parser.add_argument(
        '--random_order',
        help="Wether to yield figures in a random order.",
        action='store_false'
    )


class FigureGenerator(ABC):
    """
    Abstract class representing a figure generator.
    A FigureGenerator is a callable yielding Figure objects.

    Attributes:
        data_dir (str):                 The path to the directory where the image data sets are
                                            stored.
        default_random_order (bool):    Wether to yield figures in a random order.
    """

    def __init__(self, default_random_order: bool = False) -> None:
        self.data_dir: str = DATA_DIR
        self.default_random_order: bool = default_random_order

    @abstractmethod
    def __copy__(self) -> FigureGenerator:
        raise NotImplementedError('This method has to be implemented for each subclass.')

    @abstractmethod
    def __call__(self, random_order: Optional[bool] = None) -> Iterable[Figure]:
        """
        Abstract method that has to be implemented.
        It has to be an iterable (generator function) that yields Figure objects.

        Args:
            random_order (Optional[bool]):  Wether to yield figures in a random order.
                                                Defaults to the value given in the constructor.

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
        default_random_order (bool):    Wether to yield figures in a random order.
    """

    def __init__(
            self,
            base_figure_generator: FigureGenerator,
            function: Callable[[Figure], None],
            default_random_order: bool = True
    ) -> None:
        """
        Args:
            base_figure_generator (FigureGenerator):    A figure generator yielding Figure
                                                            objects.
            function (Callable[[Figure], None]):        A function taking a Figure as argument and
                                                            modifying it.
            default_random_order (bool):    Wether to yield figures in a random order.
        """
        super().__init__(default_random_order=default_random_order)

        self._base_figure_generator = base_figure_generator
        self._function = function

    def __copy__(self) -> StackedFigureGenerator:

        return StackedFigureGenerator(
            base_figure_generator=copy.copy(self._base_figure_generator),
            function=self._function,
            default_random_order=self.default_random_order
        )

    def __call__(self,
                 random_order: bool = None) -> Iterable[Figure]:

        for figure in self._base_figure_generator(random_order=random_order):

            self._function(figure)

            yield figure
