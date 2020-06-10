"""
TODO
"""

import os
import sys
import logging

from panel_seg.utils.figure.figure import Figure
from .figure_generator import FigureGenerator


class IphotodrawXmlFigureGenerator(FigureGenerator):
    """
    Generator of Figure objects from iPhotoDraw xml annotations.
    The input files can be provided either from a csv list or from the path
    to the directory where the image files are.
    """

    def __init__(self,
                 eval_list_txt: str = None,
                 image_directory_path: str = None):
        """
        Args:
            eval_list_txt (str):            The path of the list of figures which annotations
                                                have to be loaded.
            image_directory_path (str):     The path of the directory where the images are stored
        """

        super().__init__()

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
            logging.error(
                "Either one of `eval_list_txt` and `input_directory` options has to be set.")
            sys.exit(1)

        self.image_paths = image_paths



    def __call__(self) -> Figure:
        """
        TODO
        Yields:
            figure (Figure): Figure objects with annotations.
        """

        # Looping over the list of image paths
        for image_index, image_path in enumerate(self.image_paths):
            # print('Processing Image {}/{} : {}'.format(image_index + 1,
                                                       # num_images,
                                                       # image_path))

            # TODO remove
            # if image_index > 50:
                # return


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
