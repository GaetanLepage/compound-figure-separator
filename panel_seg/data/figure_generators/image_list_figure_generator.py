
"""
TODO
"""

import os
import logging


from panel_seg.utils.figure.figure import Figure
from .figure_generator import FigureGenerator


class ImageListFigureGenerator(FigureGenerator):
    """
    Generator of Figure objects from an image list.
    This generator does not load any annotations.
    """

    def __init__(self,
                 image_list_txt: str,
                 image_directory_path: str = None):
        """
        TODO

        Args:
            image_list_txt (str):   The path of the list of images to be loaded.

        """
        super().__init__()

        if not os.path.isfile(image_list_txt):
            raise FileNotFoundError("The evaluation list file does not exist :"\
                "\n\t {}".format(image_list_txt))

        self.image_directory_path = image_directory_path

        self.image_list_txt = image_list_txt


    def __call__(self) -> Figure:
        """
        Generator of Figure objects from a single csv annotation file.

        Yields:
            figure (Figure): Figure objects without annotations.
        """

        with open(self.image_list_txt, 'r') as image_list_file:

            for image_counter, line in enumerate(image_list_file.readlines()):

                if self.image_directory_path is not None:
                    image_file_path = os.path.join(self.image_directory_path, line[:-1])
                elif os.path.isfile(line):
                    image_file_path = line
                else:
                    image_file_path = os.path.join('data/', line)

                if not os.path.isfile(image_file_path):
                    logging.warning("File not found : %s", image_file_path)
                    continue

                figure = Figure(image_path=image_file_path,
                                index=image_counter)

                figure.load_image()

                yield figure
