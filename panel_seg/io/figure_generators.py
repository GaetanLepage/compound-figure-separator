"""
Figure objects generators from different data sets.
"""

import os
import sys
import logging
import csv
import xml.etree.ElementTree as ET

from ..utils.figure.figure import Figure
from ..utils.figure.panel import Panel


def image_clef_xml_figure_generator(
        xml_annotation_file_path: str,
        image_directory_path: str,
    ):
    """
    Generator of Figure objects from ImageCLEF data set.

    Args:
        xml_annotation_file_path:   The path of the xml annotation file
        image_directory_path (str): The path of the directory where the images are stored
    """

    # Open and parse the xml annotation file
    tree = ET.parse(xml_annotation_file_path)

    # get root element
    root = tree.getroot()

    annotation_items = root.findall('./annotation')
    for annotation_item in annotation_items:
        filename_item = annotation_item.find('./filename')
        image_filename = filename_item.text

        image_path = os.path.join(
            image_directory_path,
            image_filename + '.jpg')

        # Create Figure object
        figure = Figure(
            image_path=image_path)

        # Load image file
        figure.load_image()

        # Loop over the panels (object_items)
        panels = list()
        object_items = annotation_item.findall('./object')
        for object_item in object_items:

            point_items = object_item.findall('./point')

            # Create Panel object
            panel = Panel(
                panel_rect=[
                    point_items[0].get('x'),
                    point_items[0].get('y'),
                    point_items[3].get('x'),
                    point_items[3].get('y')
                    ]
                )

            # Add this panel to the list of panels
            panels.append(panel)

        # Store the list of panels in the Figure object
        figure.panels = panels

        yield figure


def iphotodraw_xml_figure_generator(
        eval_list_csv: str = None,
        image_directory_path: str = None):
    """
    Generator of Figure objects from iPhotoDraw xml annotations.
    The input files can be provided either from a csv list or from the path
        to the directory where the image files are.

    Args:
        eval_list_csv:              The path of the list of figures which annotations
                                        have to be loaded
        image_directory_path (str): The path of the directory where the images are stored
    """

    if eval_list_csv is not None and image_directory_path is not None:
        logging.error(
            "Both `eval_list_csv` and `input_directory` options cannot be simultaneously True.")
        sys.exit(1)

    # Get the list of image files
    image_paths = list()

    if eval_list_csv is not None:

        with open(eval_list_csv, 'w', newline='') as eval_list_csv_file:

            csv_reader = csv.reader(eval_list_csv_file, delimiter=',', quotechar='|')

            for row in csv_reader:
                # TODO
                print(row)
                image_paths.append(row)

    elif image_directory_path is not None:

        image_paths = [f for f in os.listdir(image_directory_path)
                       if f.endswith('.jpg') and os.path.isfile(
                           os.path.join(image_directory_path, f)
                           )
                       ]

    else:
        logging.error("Either one of `eval_list_csv` and `input_directory` options has to be set.")
        sys.exit(1)


    # Total number of images
    num_images = len(image_paths)

    # Looping over the list of image paths
    for image_index, image_path in enumerate(image_paths):
        print('Processing Image {}/{} : {}.'.format(
            image_index,
            num_images,
            image_path))

        # Create figure object
        figure = Figure(image_path=image_path)

        # Load image file
        figure.load_image()

        # Load annotation file
        xml_path = os.path.join(figure.image_path.replace('.jpg', '_data.xml'))
        figure.load_annotation_from_iphotodraw(xml_path)

        yield figure


def global_csv_figure_generator(
        csv_annotation_file_path: str):
    """
    TODO
    """
