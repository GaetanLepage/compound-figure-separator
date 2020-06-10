"""
TODO
"""

import os
import sys
import logging
import xml.etree.ElementTree as ET

from panel_seg.utils.figure.figure import Figure
from panel_seg.utils.figure.panel import Panel
from .figure_generator import FigureGenerator


class ImageClefXmlFigureGenerator(FigureGenerator):
    """
    TODO
    """

    def __init__(self,
                 xml_annotation_file_path: str,
                 image_directory_path: str) -> Figure:
        """
        Generator of Figure objects from ImageCLEF data set.

        Args:
            xml_annotation_file_path (str):     The path of the xml annotation file.
            image_directory_path (str):         The path of the directory where the images are
                                                    stored.

        Yields:
            figure (Figure): Figure objects with annotations.
        """

        # Open and parse the xml annotation file
        tree = ET.parse(xml_annotation_file_path)

        # get root element
        root = tree.getroot()

        self.annotation_items = root.findall('./annotation')

        # Total number of images
        self.num_images = len(self.annotation_items)

        self.image_directory_path = image_directory_path


    def __call__(self) -> Figure:

        for annotation_index, annotation_item in enumerate(self.annotation_items):
            filename_item = annotation_item.find('./filename')
            image_filename = filename_item.text

            image_path = os.path.join(
                self.image_directory_path,
                image_filename + '.jpg')

            # print('Processing Image {}/{} : {}'.format(
                # annotation_index + 1,
                # num_images,
                # image_path))

            # Create Figure object
            figure = Figure(image_path=image_path,
                            index=annotation_index)

            # Load image file
            try:
                figure.load_image()
            except FileNotFoundError as exception:
                logging.error(exception)
                continue

            # Loop over the panels (object_items)
            panels = list()
            object_items = annotation_item.findall('./object')
            for object_item in object_items:

                point_items = object_item.findall('./point')

                # Create Panel object
                panel = Panel(
                    panel_rect=[
                        int(point_items[0].get('x')),
                        int(point_items[0].get('y')),
                        int(point_items[3].get('x')),
                        int(point_items[3].get('y'))
                        ]
                    )

                # Add this panel to the list of panels
                panels.append(panel)

            # Store the list of panels in the Figure object
            figure.gt_panels = panels

            yield figure
