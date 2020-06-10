"""
TODO
"""

from panel_seg.utils.figure.figure import Figure

from .figure_generator import FigureGenerator


class IndividualCsvFigureGenerator(FigureGenerator):
    """
    TODO
    """

    def __init__(self, csv_annotation_directory: str):
        """
        Generator of Figure objects from individual csv annotation files (one per image).

        Args:
            csv_annotation_directory (str):     The path of the directory containing the csv
                                                    annotation files.

        Yields:
            figure (Figure): Figure objects with annotations.
        """
        super().__init__()

        self.csv_annotation_directory = csv_annotation_directory


    def __call__(self) -> Figure:
        # TODO implement the individual_csv_figure_generator()
        raise NotImplementedError
