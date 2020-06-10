"""
Figure objects generators from different data sets.
"""

import os
from abc import ABC, abstractmethod

import panel_seg

from ...utils.figure.figure import Figure


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
        data_dir (str): The path to the directory where the image data sets are stored.
        current_index (int):ck ck
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
            Figure: figure objects with and without annotations (and or detections).
        """
