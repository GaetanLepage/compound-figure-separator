"""
Class representing a figure.
"""

import sys
import os
import csv
import logging
import io
import PIL.Image
import hashlib

from typing import List

import xml.etree.ElementTree as ET
import tensorflow as tf

from cv2 import cv2

from .panel import Panel
from . import beam_search
from .. import box
from .. import dataset_util


class Figure:
    """
    A class for a Figure

    Attributes:
        image_path: is the path to the figure image file
        id: is the unique id to each figure
        image_orig: is the original color image
        panels: contain all panels
    """

    def __init__(self, image_path: str):
        """
        Constructor for a Figure object.

        Args:
            image_path: path to the image file
        """
        self.image_path = image_path
        self.image_filename = os.path.basename(self.image_path)
        self.image_format = os.path.splitext(self.image_filename)[-1]

        self.panels = None
        self.image = None
        self.preview_image = None
        self.image_width = 0
        self.image_height = 0


    def load_image(self):
        """
        Load the image using `self.image_path` and stores it
        in `self.image`.
        """
        # print(os.path.abspath(os.getcwd()))
        # print(self.image_path)
        # print(os.path.isdir(os.path.dirname(self.image_path)))
        # check if the image file exists
        if not os.path.isfile(self.image_path):
            raise FileNotFoundError(
                "The following image file does not exist and thus"\
                    " cannot be loaded:\n\t{}".format(self.image_path))

        # Open the file
        img = cv2.imread(self.image_path)

        # BGR image, we need to convert it to RGB image
        # self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image = img

        # Store the image size
        self.image_height, self.image_width = self.image.shape[:2]


    def load_annotation_from_csv(self, annotation_file_path: str):
        """
        Load figure annotations from the given (individual) csv file.

        Args:
            annotation_file_path
        """
        # Create empty list of panels
        panels = []

        # Open the csv file containing annotations
        with open(annotation_file_path, newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')

            # Loop over each row (panel)
            for row in csv_reader:
                panel_rect = [int(row[1]), int(row[2]), int(row[3]), int(row[4])]

                # create panel object
                panel = Panel(
                    label='',
                    panel_rect=panel_rect,
                    label_rect=None)

                panels.append(panel)

        self.panels = panels


    def export_annotation_to_individual_csv(self,
                                            csv_export_dir: str = None):
        """
        TODO
        """

        # By default the csv is at the same location
        if csv_export_dir is None:
            csv_export_dir = os.path.dirname(self.image_path)

        # check if directory exists
        if not os.path.isdir(csv_export_dir):
            logging.error(
                "Export directory does not exist : %s",
                csv_export_dir)

        # Remove extension from original figure image file name
        csv_export_file_name = os.path.splitext(
            self.image_filename)[0] + '.csv'

        csv_file_path = os.path.join(
            csv_export_dir,
            csv_export_file_name)

        # check if file already exists
        if os.path.isfile(csv_file_path):
            logging.warning(
                "The csv individual annotation file already exist :%s\n\t==> Skipping.",
                csv_file_path)
            return

        with open(csv_file_path, 'w', newline='') as csvfile:

            csv_writer = csv.writer(csvfile, delimiter=',')

            # Looping over Panel objects
            for panel in self.panels:

                csv_row = [
                    self.image_path,
                    panel.panel_rect[0],
                    panel.panel_rect[1],
                    panel.panel_rect[2],
                    panel.panel_rect[3],
                    'panel'
                    ]

                # Writting to csv file
                csv_writer.writerow(csv_row)


    def load_annotation_from_iphotodraw(self,
                                        annotation_file_path: str):
        """
        Load iPhotoDraw annotation.
        Deal with Zou's data set

        Args:
            annotation_file_path
        """

        def extract_bbox_from_iphotodraw_node(item):
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
            if x_max > self.image_width:
                x_max = self.image_width
            if y_max > self.image_height:
                y_max = self.image_height

            return x_min, y_min, x_max, y_max


        def extract_panel_info() -> List[Panel]:
            """
            Extract information from and validate all panel items

            Returns:
                panels: a list of Panel objects
            """
            panels = []
            for panel_item in panel_items:
                text_item = panel_item.find('./BlockText/Text')
                label_text = text_item.text
                label_text = label_text.strip()
                words = label_text.split(' ')

                # Panels can only have 1 or 2 words :
                # *) The first one is "panel"
                # *) The second one is the label letter
                if len(words) > 2:
                    logging.error(
                        '%s: %s is not correct',
                        annotation_file_path,
                        label_text)
                    continue

                # If the label text contains two words,
                # then the second one is the label text
                if len(words) == 2:
                    label_text = words[1]

                    if len(label_text) != 1:
                        # Now we process single character panel label only (a, b, c...)
                        logging.warning(
                            '%s: panel %s is not single character',
                            annotation_file_path,
                            label_text)

                # The text only contains a single panel
                # => no label
                else:
                    label_text = ''

                x_min, y_min, x_max, y_max = extract_bbox_from_iphotodraw_node(
                    item=panel_item)

                if x_max <= x_min or y_max <= y_min:
                    logging.error(
                        '%s: panel %s rect is not correct!',
                        annotation_file_path,
                        label_text)
                    continue

                # Create Panel object
                panel_rect = [x_min, y_min, x_max, y_max]
                panel = Panel(
                    label=label_text,
                    panel_rect=panel_rect,
                    label_rect=None)

                panels.append(panel)

            return panels


        def extract_label_info() -> List[Panel]:
            """
            Extract information from and validate all label items

            Returns:
                TODO
            """
            labels = []
            for label_item in label_items:
                text_item = label_item.find('./BlockText/Text')
                label_text = text_item.text
                label_text = label_text.strip()
                words = label_text.split(' ')
                if len(words) != 2:
                    logging.error(
                        '%s: %s is not correct',
                        annotation_file_path,
                        label_text)
                    continue
                label_text = words[1]
                # Now we process single character panel label only
                if len(label_text) != 1:
                    logging.warning(
                        '%s: label %s is not single character',
                        annotation_file_path,
                        label_text)

                x_min, y_min, x_max, y_max = extract_bbox_from_iphotodraw_node(
                    item=label_item)

                if x_max <= x_min or y_max <= y_min:
                    logging.error(
                        '%s: label %s rect is not correct!',
                        annotation_file_path,
                        label_text)
                    continue

                label_rect = [x_min, y_min, x_max, y_max]
                # We use Panel objects temporarily
                label = Panel(
                    label=label_text,
                    panel_rect=None,
                    label_rect=label_rect)

                labels.append(label)

            return labels


        def match_panels_with_labels(panels: List[Panel],
                                     labels: List[Panel]) -> List[Panel]:
            """
            Match both lists to get a unique list of panels containing
            information of their matching label

            Args:
                panels: list of panels without label information
                labels: list of labels without panel information

            Returns:
                Updated list of panels
            """
            if len(labels) != 0 and len(labels) != len(panels):
                logging.warning(
                    '%s: has different panel and label rects. Most likely there"\
                        " are mixes with-label and without-label panels',
                    annotation_file_path)

            # collect all panel label characters
            char_set = set()
            for panel in panels:
                if len(panel.label) != 0:
                    char_set.add(panel.label)

            # build panel dictionary according to labels
            panel_dict = {s: [] for s in char_set}
            for panel in panels:
                if len(panel.label) != 0:
                    panel_dict[panel.label].append(panel)

            # build label dictionary according to labels
            label_dict = {s: [] for s in char_set}
            for label in labels:
                label_dict[label.label].append(label)

            # assign labels to panels
            for label_char in char_set:
                if len(panel_dict[label_char]) != len(label_dict[label_char]):
                    logging.error(
                        '%s: panel %s does not have same matching labels!',
                        annotation_file_path,
                        label_char)
                    continue

                # TODO why do we use Beam search here ? We are supposed to
                #   be dealing with ground truth
                # Beam search algorithm to map labels to panels
                beam_search.assign_labels_to_panels(panels=panel_dict[label_char],
                                                    labels=label_dict[label_char])

            # expand the panel_rect to always include label_rect
            for panel in panels:
                if panel.label_rect is not None:
                    panel.panel_rect = box.union(rectangle_1=panel.label_rect,
                                                 rectangle_2=panel.panel_rect)

            return panels


        # create element tree object
        tree = ET.parse(annotation_file_path)

        # get root element
        root = tree.getroot()

        shape_items = root.findall('./Layers/Layer/Shapes/Shape')

        # Read All Items (Panels and Labels)
        panel_items = []
        label_items = []
        for shape_item in shape_items:
            text_item = shape_item.find('./BlockText/Text')
            text = text_item.text.lower()
            if text.startswith('panel'):
                panel_items.append(shape_item)
            elif text.startswith('label'):
                label_items.append(shape_item)
            else:
                logging.error(
                    '%s: has unknown <shape> xml items %s',
                    annotation_file_path,
                    text)

        # Extract information from and validate all panel items
        panels = extract_panel_info()

        # Extract information from and validate all label items
        labels = extract_label_info()

        # Match both lists to get a unique list of panels containing
        #   information of their matching label
        # Save this list of panels in tha appropriate attribute of
        #   the figure object
        self.panels = match_panels_with_labels(panels=panels,
                                               labels=labels)


    def get_preview(self, force=False):
        """
        Generate an image preview for the figure.
        It consists in drawing the panels (and labels, if applicable) bounding boxes
        on top of the image.

        Args:
            force: if True, forces the preview image to be recomputed even if
                the corresponding attribute was not empty.

        Returns:
            preview_img: the preview image
        """
        if self.preview_image is not None and not force:
            return self.preview_image

        shape_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        if self.image is None:
            self.load_image()

        preview_img = self.image.copy()

        # Drowing rectangles
        for panel_index, panel in enumerate(self.panels):

            # Select color
            color = shape_colors[panel_index % len(shape_colors)]

            # Draw panel box
            cv2.rectangle(img=preview_img,
                          pt1=(panel.panel_rect[0], panel.panel_rect[1]),
                          pt2=(panel.panel_rect[2], panel.panel_rect[3]),
                          color=color,
                          thickness=3)

            if panel.label_rect is not None:
                # Draw label box
                cv2.rectangle(img=preview_img,
                              pt1=(panel.label_rect[0], panel.label_rect[1]),
                              pt2=(panel.label_rect[2], panel.label_rect[3]),
                              color=color,
                              thickness=2)

                # Draw label text
                cv2.putText(img=preview_img,
                            text=panel.label,
                            org=(panel.label_rect[2] + 10, panel.label_rect[3]),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=color)

        # Store the computed image
        self.preview_image = preview_img

        return preview_img


    def show_preview(self, delay=0, window_name=None):
        """
        TODO

        Args:
            delay:       The number of seconds after which the window is closed
                if 0, the delay is disabled.
            window_name: Name of the image display window.
        """

        image_preview = self.get_preview()

        if window_name is None:
            window_name = self.image_filename

        cv2.imshow(window_name, image_preview)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()


    def save_preview(self, folder):
        """
        Save the annotation preview at folder.
        """
        preview_img = self.get_preview()


        # Remove extension from original figure image file name
        file_name = os.path.splitext(
            self.image_filename)[0]

        export_path = os.path.join(folder, file_name + "_preview.jpg")

        # Write the preview image file to destination
        cv2.imwrite(export_path, preview_img)


    def convert_to_tf_example(self):
        """
        Convert the figure to a TensorFlow example which is compatible with the TensorFlow
        Object Detection API.

        Returns:
            example: The corresponding tf example.
        """

        # Load image
        with tf.io.gfile.GFile(self.image_path, 'rb') as fid:
            encoded_jpg = fid.read()

        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)

        # Generate unique id
        key = hashlib.sha256(encoded_jpg).hexdigest()

        # Check that image shape is correct
        assert image.size[0] == self.image_width, "Inconsistent image width."
        assert image.size[1] == self.image_height, "Inconsistent image height."

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        for panel in self.panels:
            # Bounding box
            xmin.append(float(panel.panel_rect[0]) / self.image_width)
            ymin.append(float(panel.panel_rect[1]) / self.image_height)
            xmax.append(float(panel.panel_rect[2]) / self.image_width)
            ymax.append(float(panel.panel_rect[3]) / self.image_height)

            # Class information
            class_name = 'panel'
            classes_text.append(class_name.encode('utf8'))
            classes.append(1)

        feature = {
            'image/key/sha256':
                dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded':
                dataset_util.bytes_feature(encoded_jpg),
            'image/source_id':
                dataset_util.bytes_feature(self.image_filename.encode('utf8')),
            'image/filename':
                dataset_util.bytes_feature(self.image_filename.encode('utf8')),
            'image/height':
                dataset_util.int64_feature(self.image_height),
            'image/width':
                dataset_util.int64_feature(self.image_width),
            'image/format':
                dataset_util.bytes_feature(self.image_format.encode('utf8')),
            'image/object/bbox/xmin':
                dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax':
                dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin':
                dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax':
                dataset_util.float_list_feature(ymax),
            'image/object/class/text':
                dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label':
                dataset_util.int64_list_feature(classes),
            }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        return example
