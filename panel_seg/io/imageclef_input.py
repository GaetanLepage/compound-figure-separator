"""
Deal with annotations from ImageCLEF xml annotation file.
"""

from .figure_generators import image_clef_xml_figure_generator
from .figure_viewer import view_data_set
from ..utils.figure.misc import export_figures_to_csv



def convert_annotations_imageclef_to_csv(
        xml_annotation_file_path: str = None,
        image_directory_path: str = None,
        output_csv_file: str = None):
    """
    Convert annotations from ImageCLEF xml annotation file to a CSV annotations file.

    Args:
        xml_annotation_file_path:   The path of the xml annotation file
        image_directory_path (str): The path of the directory where the images are stored
        output_csv_file (str):      The path where to store the newly created csv file
    """

    figure_generator = image_clef_xml_figure_generator(
        xml_annotation_file_path=xml_annotation_file_path,
        image_directory_path=image_directory_path
        )

    export_figures_to_csv(
        figure_generator=figure_generator,
        output_csv_file=output_csv_file)


def show_imageclef_data_set(
        xml_annotation_file_path: str = None,
        image_directory_path: str = None):
    """
    Preview all the figures from an ImageCLEF data set.

    Args:
        xml_annotation_file_path:   The path of the xml annotation file
        image_directory_path (str): The path of the directory where the images are stored
    """

    figure_generator = image_clef_xml_figure_generator(
        xml_annotation_file_path=xml_annotation_file_path,
        image_directory_path=image_directory_path)

    view_data_set(
        figure_generator=figure_generator,
        delay=100,
        window_name="ImageCLEF data preview")
