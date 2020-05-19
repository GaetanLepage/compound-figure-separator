"""
Figure objects generators from different data sets.
"""

import os
import sys
import logging
import csv
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


def image_list_figure_generator(image_list_txt: str):
    """
    Generator of Figure objects from an image list.
    This generator does not load any annotations.

    Args:
        image_list_txt:              The path of the list of images to be loaded
    """

    with open(image_list_txt, 'r') as image_list_file:

        image_counter = 0

        for line in image_list_file.readlines():

            if os.path.isfile(line):
                image_file_path = line
            else:
                image_file_path = os.path.join('data/', image_file_path)

            if not os.path.isfile(image_file_path):
                logging.warning("File not found : %s", line)
                continue

            figure = Figure(image_path=image_file_path,
                            index=image_counter)

            yield figure

            image_counter += 1



def image_clef_xml_figure_generator(xml_annotation_file_path: str,
                                    image_directory_path: str):
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

    counter = 0

    for annotation_index, annotation_item in enumerate(annotation_items):
        filename_item = annotation_item.find('./filename')
        image_filename = filename_item.text

        image_path = os.path.join(
            image_directory_path,
            image_filename + '.jpg')

        # print('Processing Image {}/{} : {}'.format(
            # annotation_index + 1,
            # num_images,
            # image_path))

        # Create Figure object
        figure = Figure(image_path=image_path,
                        index=counter)

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

        counter += 1


def iphotodraw_xml_figure_generator(eval_list_txt: str = None,
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

        # TODO remove ?
        # image_paths = [
            # os.path.abspath(line) if os.path.isfile(line)
            # else os.path.abspath(os.path.join(DATA_DIR, line))
            # for line in eval_list_lines]

        image_paths = [
            line if os.path.isfile(line)
            else os.path.join('data/', line)
            for line in eval_list_lines]

    elif image_directory_path is not None:

        image_paths = [f for f in os.listdir(image_directory_path)
                       if f.endswith('.jpg') and os.path.isfile(
                           os.path.join(image_directory_path, f)
                           )
                       ]

    else:
        logging.error("Either one of `eval_list_txt` and `input_directory` options has to be set.")
        sys.exit(1)


    # Looping over the list of image paths
    for image_index, image_path in enumerate(image_paths):
        # print('Processing Image {}/{} : {}'.format(image_index + 1,
                                                   # num_images,
                                                   # image_path))
        # Create figure object
        figure = Figure(image_path=image_path,
                        index=image_index)

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
    Generator of Figure objects from a single csv annotation file.

    Args:
        csv_annotation_file:    The path of the csv annotation file to load.
    """

    if not os.path.isfile(csv_annotation_file_path):
        raise FileNotFoundError("The prediction annotation csv file does not exist :"\
            "\n\t {}".format(csv_annotation_file_path))

    with open(csv_annotation_file_path, 'r') as csv_annotation_file:
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


def individual_csv_figure_generator(
        csv_annotation_directory: str):
    """
    Generator of Figure objects from individual csv annotation files (one per image).

    Args:
        csv_annotation_directory:   The path of the directory containing the csv annotation files.
    """

    # TODO implement the individual_csv_figure_generator()
    raise NotImplementedError
