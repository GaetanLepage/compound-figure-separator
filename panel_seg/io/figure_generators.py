"""
Figure objects generators from different data sets.
"""

import os
import sys
import logging
import xml.etree.ElementTree as ET

import panel_seg

from ..utils.figure.figure import Figure
from ..utils.figure.panel import Panel

PROJECT_DIR = os.path.join(
    os.path.dirname(panel_seg.__file__),
    os.pardir)
DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data/")


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

    # Total number of images
    num_images = len(annotation_items)

    for annotation_index, annotation_item in enumerate(annotation_items):
        filename_item = annotation_item.find('./filename')
        image_filename = filename_item.text

        image_path = os.path.join(
            image_directory_path,
            image_filename + '.jpg')

        print('Processing Image {}/{} : {}'.format(
            annotation_index + 1,
            num_images,
            image_path))

        # Create Figure object
        figure = Figure(
            image_path=image_path)

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
        figure.panels = panels

        yield figure


def iphotodraw_xml_figure_generator(
        eval_list_txt: str = None,
        image_directory_path: str = None):
    """
    Generator of Figure objects from iPhotoDraw xml annotations.
    The input files can be provided either from a csv list or from the path
        to the directory where the image files are.

    Args:
        eval_list_txt:              The path of the list of figures which annotations
                                        have to be loaded
        image_directory_path (str): The path of the directory where the images are stored
    """

    if eval_list_txt is not None and image_directory_path is not None:
        logging.error(
            "Both `eval_list_txt` and `input_directory` options cannot be simultaneously True.")
        sys.exit(1)

    # Get the list of image files
    image_paths = list()

    if eval_list_txt is not None:

        # Read list of image files
        with open(eval_list_txt, 'r') as eval_list_file:
            eval_list_lines = eval_list_file.read().splitlines()

        image_paths = [
            os.path.abspath(line) if os.path.isfile(line)
            else os.path.abspath(
                os.path.join(
                    DATA_DIR,
                    line)
                )
            for line in eval_list_lines
                ]

    elif image_directory_path is not None:

        image_paths = [f for f in os.listdir(image_directory_path)
                       if f.endswith('.jpg') and os.path.isfile(
                           os.path.join(image_directory_path, f)
                           )
                       ]

    else:
        logging.error("Either one of `eval_list_txt` and `input_directory` options has to be set.")
        sys.exit(1)


    # Total number of images
    num_images = len(image_paths)

    # Looping over the list of image paths
    for image_index, image_path in enumerate(image_paths):
        print('Processing Image {}/{} : {}'.format(
            image_index + 1,
            num_images,
            image_path))

        # Create figure object
        figure = Figure(image_path=image_path)

        try:
            # Load image file
            figure.load_image()
        except FileNotFoundError as exception:
            logging.error(exception)
            continue

        # Load annotation file
        xml_path = os.path.join(figure.image_path.replace('.jpg', '_data.xml'))
        figure.load_annotation_from_iphotodraw(xml_path)

        yield figure


def global_csv_figure_generator(
        csv_annotation_file_path: str):
    """
    TODO
    """

    # TODO
