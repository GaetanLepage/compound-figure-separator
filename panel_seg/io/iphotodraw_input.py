"""
Deal with annotations from iPhotoDraw xml annotation files.
"""

from .figure_generators import iphotodraw_xml_figure_generator
from .figure_viewer import view_data_set
from ..utils.figure.misc import export_figures_to_csv



def convert_annotations_iphotodraw_to_csv(
        eval_list_txt: str = None,
        image_directory_path: str = None,
        output_csv_file: str = None):
    """
    Convert annotations from individual xml annotation files from iPhotoDraw
        to a CSV annotations file.
    The input files can be provided either from a txt list or from the path
        to the directory where the image files are.

    Args:
        eval_list_txt:              The path of the list of figures which annotations
                                        have to be loaded
        image_directory_path (str): The path of the directory where the images are stored
        output_csv_file (str):      The path where to store the newly created csv file
    """

    figure_generator = iphotodraw_xml_figure_generator(
        eval_list_txt=eval_list_txt,
        image_directory_path=image_directory_path
        )

    export_figures_to_csv(
        figure_generator=figure_generator,
        output_csv_file=output_csv_file)


def show_iphotodraw_data_set(
        eval_list_txt: str = None,
        image_directory_path: str = None):
    """
    Preview all the figures from an iPhotoDraw data set.
    The input files can be provided either from a txt list or from the path
        to the directory where the image files are.

    Args:
        eval_list_txt:              The path of the list of figures which annotations
                                        have to be loaded
        image_directory_path (str): The path of the directory where the images are stored
    """

    figure_generator = iphotodraw_xml_figure_generator(
        eval_list_txt=eval_list_txt,
        image_directory_path=image_directory_path)

    view_data_set(
        figure_generator=figure_generator,
        delay=100,
        window_name="PanelSeg data preview")
