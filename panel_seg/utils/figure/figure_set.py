"""
DEPRECATED
"""

import os
import sys
import logging

from typing import List

from ..figure.figure import Figure
from ..figure import misc

from ..figure.panel import PanelSegError

class FigureSet:
    """
    A class for a FigureSet

    Attributes:
        file_list: TODO
        files: TODO
    """

    def __init__(self):
        """
        Constructor for a FigureSet.
        """

        self.file_list = None
        self.files = []


    def load_list(self, list_file):
        """
        Read the provided list of files and place those in
        `self.files`.

        Args:
            list_file: The path to the list of files.
        """
        self.file_list = list_file
        # self.files = misc.read_sample_list(list_file)
        self.figures = dict()


    def read_sample_list(self, list_path: List[str]):
        """
        TODO

        Args :
            list_path (TODO): TODO

        Returns :
            * TODO (TODO): TODO
        """
        # TODO
        # with tf.gfile.GFile(list_path) as fid:
            # lines = fid.readlines()
        # return [line.strip().split(' ')[0] for line in lines]


    def load_annotations_from_xml(
            self,
            xml_annotation_file_path: str):
        """
        TODO

        Args:
            xml_annotation_file_path: TODO
        """
        try:
            with open(xml_annotation_file_path, 'r') as xml_annotation_file:
                lines = xml_annotation_file.readlines()

        except FileNotFoundError:
            logging.error(
                "The following xml annotation file was not found :\n\t%s",
                xml_annotation_file_path)
            sys.exit(1)



        for line in lines:
            # TODO
            image_name = None
            image_path = None
            self.figures[image_name] = Figure(image_path)


    def load_annotations_from_iphotodraw_xml(self):
        """
        TODO
        """
        for file_index, file in enumerate(self.files):
            logging.info(
                'Validate Annotation of Image %d/%d: %s.',
                file_index, len(self.files), file)

            try:
                # Check if image file exists
                if not os.path.exists(file):
                    raise PanelSegError('Could not find {}.'.format(file))

                # Create a Figure object
                figure = Figure(file)
                # Load associated image
                figure.load_image()

                # Check if xml annotation file exists
                xml_path = os.path.join(figure.image_path.replace('.jpg', '_data.xml'))
                if not os.path.exists(xml_path):
                    raise FileNotFoundError('Could not find {}.'.format(xml_path))

                # Load annotations (panels + labels) from iPhotoDraw xml
                figure.load_annotation_from_iphotodraw(xml_path)

            except FileNotFoundError as file_not_found_error:
                logging.warning(file_not_found_error)
                continue


    def save_gt_preview(self):
        """
        Generate ground truth annotation preview for image.
        It is saved to the `preview` folder in the same directory than the image.
        """
        for idx, file in enumerate(self.files):
            # if idx != 5:
            #     continue
            logging.info('Generate GT Annotation Preview for Image %d: %s.', idx, file)

            try:
                if not os.path.exists(file):
                    raise PanelSegError('Could not find {}.'.format(file))
                figure = Figure(file)
                figure.load_image()

                xml_path = os.path.join(figure.image_path.replace('.jpg', '_data.xml'))
                if not os.path.exists(xml_path):
                    raise PanelSegError('Could not find {}.'.format(xml_path))
                figure.get_annotation_from_iphotodraw(xml_path)

            except PanelSegError as panel_seg_error:
                logging.warning(panel_seg_error)
                continue

            # Save preview
            folder, file = os.path.split(figure.image_path)
            folder = os.path.join(folder, "preview")
            figure.save_preview(folder)


    def convert_to_csv(self):
        """
        TODO
        """
