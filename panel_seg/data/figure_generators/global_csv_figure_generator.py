"""
TODO
"""

import os
import logging
import csv

from panel_seg.utils.figure.panel import Panel

from panel_seg.utils.figure.figure import Figure
from .figure_generator import FigureGenerator


class GlobalCsvFigureGenerator(FigureGenerator):
    """
    TODO
    """

    def __init__(self, csv_annotation_file_path: str):
        """
        Init function.
        Call the init function of the abstract parent class.

        Args:
            csv_annotation_file_path (str): The path of the csv annotation file.
        """

        self.csv_annotation_file_path = csv_annotation_file_path

        super().__init__()

        if not os.path.isfile(csv_annotation_file_path):
            raise FileNotFoundError("The prediction annotation csv file does not exist :"\
                "\n\t {}".format(csv_annotation_file_path))


    def __call__(self) -> Figure:
        """
        Generator of Figure objects from a single csv annotation file.

        Args:
            csv_annotation_file (str):  The path of the csv annotation file to load.

        Yields:
            figure (Figure): Figure objects with annotations.
        """


        with open(self.csv_annotation_file_path, 'r') as csv_annotation_file:
            csv_reader = csv.reader(csv_annotation_file, delimiter=',')

            panels = []
            image_path = ''
            figure = None

            image_counter = 0

            for row in csv_reader:

                # New figure
                if not image_path.endswith(row[0]):
                    if figure is not None:
                        figure.gt_panels = panels
                        yield figure

                        image_counter += 1

                    image_path = row[0]
                    if not os.path.isfile(image_path):
                        image_path = os.path.join('data/', image_path)
                    if not os.path.isfile(image_path):
                        raise FileNotFoundError("The following image file does not exist :"\
                            "\n\t {}".format(image_path))

                    figure = Figure(image_path=image_path,
                                    index=image_counter)
                    # Load image file
                    try:
                        figure.load_image()
                    except FileNotFoundError as exception:
                        logging.error(exception)
                        continue

                    panels = []


                # Panel segmentation + panel splitting
                if len(row) == 11:
                    try:
                        label_coordinates = [int(x) for x in row[6:10]]
                        label = row[10]
                    except ValueError:
                        label_coordinates = None
                        label = None

                # Panel splitting only
                elif len(row) == 6:
                    label_coordinates = None
                    label = None
                else:
                    raise ValueError("Row should be of length 6 or 11")

                panel_coordinates = [int(x) for x in row[1:5]]
                panel_class = row[5]
                assert panel_class == 'panel'

                # Instanciate Panel object
                panel = Panel(panel_rect=panel_coordinates,
                              label_rect=label_coordinates,
                              label=label)

                panels.append(panel)

            # set panels for the last figure
            figure.gt_panels = panels
            yield figure
