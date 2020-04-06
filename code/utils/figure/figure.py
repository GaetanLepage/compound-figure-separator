"""
Class representing a figure.
"""


import os
import csv
import logging

from typing import List

import xml.etree.ElementTree as ET

from cv2 import cv2

from panel import Panel
from figure import misc
from figure import box
# import _figure_zou



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
        self.panels = None
        self.image = None
        self.image_width = 0
        self.image_height = 0


    def load_image(self):
        """
        Load the image using `self.image_path` and stores it
        in `self.image`.
        """

        # Open the file
        img = cv2.imread(self.image_path)

        # BGR image, we need to convert it to RGB image
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Store the image size
        self.image_height, self.image_width = self.image.shape[:2]


    def load_annotation_from_csv(self, annotation_file_path: str):
        """
        Load figure annotations from the given csv file.

        Args:
            annotation_file_path
        """
        # Create empty list of panels
        panels = []

        # Open the csv file containing annotations
        with open(annotation_file_path, newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')

            # Loop over each low (panel)
            for row in csv_reader:
                panel_rect = [int(row[1]), int(row[2]), int(row[3]), int(row[4])]

                # create panel object
                panel = Panel(
                    label='',
                    panel_rect=panel_rect,
                    label_rect=None)

                panels.append(panel)

        self.panels = panels


    def get_annotation_from_iphotodraw(
            self,
            annotation_file_path: str):
        """
        Load iPhotoDraw annotation.
        Deal with Zou's data set

        Args:
            annotation_file_path
        """

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

                x_min, y_min, x_max, y_max = misc.extract_bbox_from_iphotodraw_node(
                    item=panel_item,
                    image_width=self.image_width,
                    image_height=self.image_height)

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

                x_min, y_min, x_max, y_max = misc.extract_bbox_from_iphotodraw_node(
                    item=label_item,
                    image_width=self.image_width,
                    image_height=self.image_height)

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


        def match_panels_with_labels(
                panels: List[Panel],
                labels: List[Panel]
                ) -> List[Panel]:
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
                misc.assign_labels_to_panels(
                    panels=panel_dict[label_char],
                    labels=label_dict[label_char])

            # expand the panel_rect to always include label_rect
            for panel in panels:
                if panel.label_rect is not None:
                    panel.panel_rect = box.union(
                        rectangle_1=panel.label_rect,
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
        self.panels = match_panels_with_labels(
            panels=panels,
            labels=labels)

    def save_preview(self, folder):
        """
        Save the annotation preview at folder.
        """
        shape_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        _, file = os.path.split(self.image_path)

        export_path = os.path.join(folder, file)
        preview_img = self.image.copy()

        # Drowing rectangles
        for panel_index, panel in enumerate(self.panels):

            # Select color
            color = shape_colors[panel_index % len(shape_colors)]
            cv2.rectangle(
                img=preview_img,
                pt1=(panel.panel_rect[0], panel.panel_rect[1]),
                pt2=(panel.panel_rect[2], panel.panel_rect[3]),
                color=color,
                thickness=3)

            if panel.label_rect is not None:
                cv2.rectangle(
                    img=preview_img,
                    pt1=(panel.label_rect[0], panel.label_rect[1]),
                    pt2=(panel.label_rect[2], panel.label_rect[3]),
                    color=color,
                    thickness=2)
                cv2.putText(
                    img=preview_img,
                    text=panel.label,
                    org=(panel.label_rect[2] + 10, panel.label_rect[3]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=color)

        cv2.imwrite(export_path, preview_img)
