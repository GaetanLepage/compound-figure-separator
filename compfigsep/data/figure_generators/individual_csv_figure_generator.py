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


#####################################################
Figure generator for individual csv annotation files.
"""

from typing import Iterable

from ...utils.figure.figure import Figure
from .figure_generator import FigureGenerator


class IndividualCsvFigureGenerator(FigureGenerator):
    """
    Generator of Figure objects from individual csv annotation files (one per image).

    Attributes:
        data_dir (str):                 The path to the directory where the image data sets are
                                            stored.
        current_index (int):            Index of the currently handled figure. This helps knowing
                                            the "progression" of the data loading process.
        csv_annotation_directory (str): The path of the directory containing the csv annotation
                                            files.
    """

    def __init__(self,
                 csv_annotation_directory: str
                 ) -> None:
        """
        Init for IndividualCsvFigureGenerator.

        Args:
            csv_annotation_directory (str):     The path of the directory containing the csv
                                                    annotation files.
        """
        super().__init__()

        self.csv_annotation_directory = csv_annotation_directory


    def __call__(self) -> Iterable[Figure]:
        """
        Returns:
            Iterable[Figure]:   Yields Figure objects with annotations.
        """
        # TODO implement the call method for the IndividualCsvFigureGenerator.
        raise NotImplementedError
