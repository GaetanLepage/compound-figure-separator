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
Figure generator handling the ImageCLEF data set.
"""

import os
import logging
import xml.etree.ElementTree as ET

from ...utils.figure.figure import Figure
from ...utils.figure.panel import Panel
from .figure_generator import FigureGenerator


class ImageClefXmlFigureGenerator(FigureGenerator):
    """
    Generator of Figure objects from the ImageCLEF data set.

    Attributes:
        data_dir (str):                         The path to the directory where the image data
                                                    sets are stored.
        current_index (int):                    Index of the currently handled figure. This helps
                                                    knowing the "progression" of the data loading
                                                    process.
        annotation_items (List[ET.Element]):    List of annotations for the whole data set.
        num_images (int):                       The number of images in the data set.
        image_directory_path (str):             Path to the directory where image files are
                                                    stored.
    """

    def __init__(self,
                 xml_annotation_file_path: str,
                 image_directory_path: str) -> Figure:
        """
        Generator of Figure objects from ImageCLEF data set.

        Args:
            xml_annotation_file_path (str): The path of the xml annotation file.
            image_directory_path (str):     The path of the directory where the images are stored.

        Yields:
            figure (Figure): Figure objects with annotations.
        """

        # Open and parse the xml annotation file.
        tree = ET.parse(xml_annotation_file_path)

        # Get root element.
        root = tree.getroot()

        # Get the annotation data from the parsed xml.
        self.annotation_items = root.findall('./annotation')

        # Total number of images
        self.num_images = len(self.annotation_items)

        # path of the directory where the images are stored.
        self.image_directory_path = image_directory_path


    def __call__(self) -> Figure:
        """
        'Generator' method yielding annotated figures from the ImageCLEF data set.

        Yields:
            figure (Figure): Figure objects with annotations.
        """

        # Loop over the annotation items.
        for annotation_index, annotation_item in enumerate(self.annotation_items):

            # Image filename
            filename_item = annotation_item.find('./filename')
            image_filename = filename_item.text

            # Image path
            image_path = os.path.join(
                self.image_directory_path,
                image_filename + '.jpg')

            # TODO maybe set up a verbose mode
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
            panels = []
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
