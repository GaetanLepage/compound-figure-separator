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


#################################################
Figure generator handling the ImageCLEF data set.
"""

from __future__ import annotations
import os
import logging
import random
import xml.etree.ElementTree as ET
from typing import cast, Iterable, List, Optional
from argparse import ArgumentParser

from ...utils.figure import Figure, Panel, SubFigure
from ...utils.box import Box
from .figure_generator import FigureGenerator


def add_image_clef_args(parser: ArgumentParser) -> None:
    """
    Parse the argument for loading an ImageCLEF annotation file.

    Args:
        parser (ArgumentParser):        An ArgumentParser.
    """

    parser.add_argument('--annotation_xml',
                help="The path to the xml annotation file.",
                default="data/ImageCLEF/training/FigureSeparationTraining2016GT.xml",
                type=str)

    parser.add_argument('--image_directory_path',
                        help="The path to the directory where the images are stored.",
                        default="data/ImageCLEF/training/FigureSeparationTraining2016/",
                        type=str)


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
        default_random_order (bool):    Wether to yield figures in a random order.
    """

    def __init__(self,
                 xml_annotation_file_path: str,
                 image_directory_path: str,
                 default_random_order: bool = False) -> None:
        """
        Generator of Figure objects from ImageCLEF data set.

        Args:
            xml_annotation_file_path (str): The path of the xml annotation file.
            image_directory_path (str):     The path of the directory where the images are stored.
            default_random_order (bool):    Wether to yield figures in a random order.

        Yields:
            figure (Figure): Figure objects with annotations.
        """
        super().__init__(default_random_order=default_random_order)

        self.xml_annotation_file_path: str = xml_annotation_file_path

        # Open and parse the xml annotation file.
        tree: ET.ElementTree = ET.parse(xml_annotation_file_path)

        # Get root element.
        root: ET.Element = tree.getroot()

        # Get the annotation data from the parsed xml.
        self.annotation_items = root.findall('./annotation')

        # Total number of images
        self.num_images = len(self.annotation_items)

        # path of the directory where the images are stored.
        self.image_directory_path = image_directory_path


    def __copy__(self) -> ImageClefXmlFigureGenerator:

        return ImageClefXmlFigureGenerator(xml_annotation_file_path=self.xml_annotation_file_path,
                                           image_directory_path=self.image_directory_path,
                                           default_random_order=self.default_random_order)


    def __call__(self, random_order: Optional[bool] = None) -> Iterable[Figure]:
        """
        'Generator' method yielding annotated figures from the ImageCLEF data set.

        Returns:
            Iterable[Figure]:   Figure objects with annotations.
        """
        if random_order is None:
            random_order = self.default_random_order

        annotation_items_copy = self.annotation_items.copy()

        if random_order:
            random.shuffle(annotation_items_copy)

        # Loop over the annotation items.
        for annotation_index, annotation_item in enumerate(annotation_items_copy):

            # Image filename
            filename_item = annotation_item.find('./filename')

            if filename_item is None or filename_item.text is None:
                continue

            image_filename: str = filename_item.text

            # Image path
            image_path = os.path.join(
                self.image_directory_path,
                image_filename + '.jpg')

            # Create Figure object
            figure: Figure = Figure(image_path=image_path,
                                    index=annotation_index)

            # Load image file
            try:
                figure.load_image()
            except FileNotFoundError as exception:
                logging.error(exception)
                continue

            # Loop over the panels (object_items)
            subfigures = []
            object_items = annotation_item.findall('./object')
            for object_item in object_items:

                point_items = object_item.findall('./point')

                coordinates = [point_items[0].get('x'),
                               point_items[0].get('y'),
                               point_items[3].get('x'),
                               point_items[3].get('y')]

                if any([coord is None for coord in coordinates]):
                    continue

                str_coordinates: List[str] = cast(List[str],
                                                  coordinates)

                # Create Panel object
                panel: Panel = Panel(box=cast(Box,
                                              (int(coord)
                                               for coord in str_coordinates)))

                # Add this panel to the list of panels
                subfigures.append(SubFigure(panel=panel))

            # Store the list of panels in the Figure object
            figure.gt_subfigures = subfigures

            yield figure
