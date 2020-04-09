"""
Deal with annotations from iPhotoDraw xml annotation files.
"""

from .figure_generators import iphotodraw_xml_figure_generator
from .figure_viewer import view_data_set
from ..utils.figure.misc import export_figures_to_csv

def extract_bbox_from_iphotodraw_node(item, image_width, image_height):
    """
    Extract bounding box information from Element item (ElementTree).
    It also makes sure that the bounding box is within the image.

    Args:
        item (Element item): Either a panel or label item extracted from
                                an iPhotoDraw xml annotation file.
        image_width (int):   The width of the image
        image_height (int):  The height of the image

    Returns:
        return (x_min, y_min, x_max, y_max): The coordinates of the bounding box
    """
    extent_item = item.find('./Data/Extent')

    # Get data from the xml item
    height_string = extent_item.get('Height')
    width_string = extent_item.get('Width')

    x_string = extent_item.get('X')
    y_string = extent_item.get('Y')

    # Compute coordinates of the bounding box
    x_min = round(float(x_string))
    y_min = round(float(y_string))
    x_max = x_min + round(float(width_string))
    y_max = y_min + round(float(height_string))

    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_max > image_width:
        x_max = image_width
    if y_max > image_height:
        y_max = image_height

    return x_min, y_min, x_max, y_max


def convert_annotations_iphotodraw_to_csv(
        eval_list_csv: str = None,
        image_directory_path: str = None,
        output_csv_file: str = None):
    """
    Convert annotations from individual xml annotation files from iPhotoDraw
        to a CSV annotations file.
    The input files can be provided either from a csv list or from the path
        to the directory where the image files are.

    Args:
        eval_list_csv:              The path of the list of figures which annotations
                                        have to be loaded
        image_directory_path (str): The path of the directory where the images are stored
        output_csv_file (str):      The path where to store the newly created csv file
    """

    figure_generator = iphotodraw_xml_figure_generator(
        eval_list_csv=eval_list_csv,
        image_directory_path=image_directory_path
        )

    export_figures_to_csv(
        figure_generator=figure_generator,
        output_csv_file=output_csv_file)


def show_iphotodraw_data_set(
        eval_list_csv: str = None,
        image_directory_path: str = None):
    """
    Preview all the figures from an iPhotoDraw data set.
    The input files can be provided either from a csv list or from the path
        to the directory where the image files are.

    Args:
        eval_list_csv:              The path of the list of figures which annotations
                                        have to be loaded
        image_directory_path (str): The path of the directory where the images are stored
    """

    figure_generator = iphotodraw_xml_figure_generator(
        eval_list_csv=eval_list_csv,
        image_directory_path=image_directory_path)

    view_data_set(figure_generator=figure_generator)
