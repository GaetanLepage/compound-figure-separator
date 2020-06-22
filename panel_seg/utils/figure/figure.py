"""
#############################
#        CompFigSep         #
# Compound Figure Separator #
#############################

GitHub:         https://github.com/GaetanLepage/compound-figure-separator

Author:         Gaétan Lepage
Email:          gaetan.lepage@grenoble-inp.org
Date:           Spring 2020

Master's project @HES-SO (Sierre, SW)

Supervisors:    Henning Müller (henning.mueller@hevs.ch)
                Manfredo Atzori (manfredo.atzori@hevs.ch)

Collaborator:   Niccolò Marini (niccolo.marini@hevs.ch)


############################
Class representing a figure.
"""

import os
import csv
import logging
import io
import hashlib
from typing import List, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import PIL.Image
import tensorflow as tf
from cv2 import cv2

from .panel import Panel
from . import beam_search
from .. import box
from .. import dataset_util
from . import label_class


class Figure:
    """
    A class representing a Figure (and eventually ground truth and/or predicted annotations).

    Attributes:
        index (int):                            The unique id to each figure.
        image_path (str):                       The path to the figure image file.
        image_filename (str):                   Image file name.
        image_format (str):                     The format of the image file (jpg, png,...).
        image (np.ndarray):                     The image data.
        image_width (int):                      The image width (in pixels).
        image_height (int):                     The image height (in pixels).
        preview_image (np.ndarray):             The preview image (image + bounding boxes)
        gt_panels (List[Panel]):                Ground truth panel objects.
        detected_panels (List[DetectedPanel]):  Detected panel objects.
    """

    def __init__(self,
                 image_path: str,
                 index: int):
        """
        Init for a Figure object.
        Neither the image or the annotations are loaded at this stage.

        Args:
            image_path (str): The path to the figure image file.
            index (int):      A unique index to identify the figure within the data set.
        """

        # Figure identifier
        self.index = index

        # Image information
        self.image_path = image_path
        self.image_filename = os.path.basename(self.image_path)
        self.image_format = os.path.splitext(self.image_filename)[-1]

        self.image = None
        self.image_width = 0
        self.image_height = 0


        # Can contain a preview of the image with its bounding boxes.
        self.preview_image = None

        # Ground truth panel annotations
        self.gt_panels = None

        # Predicted panels
        self.detected_panels = None

        # Logger
        self._logger = logging.getLogger(__name__)


    def load_image(self):
        """
        Load the image using `self.image_path` and stores it in `self.image`.
        """

        # No need to reload the image if it has already been done.
        if self.image is not None:
            return

        # check if the image file exists
        if not os.path.isfile(self.image_path):
            raise FileNotFoundError("The following image file does not exist and thus"\
                                    " cannot be loaded:\n\t{}".format(self.image_path))

        # Open the file
        img = cv2.imread(self.image_path)

        # BGR image, we need to convert it to RGB image
        # self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image = img

        # Store the image size
        self.image_height, self.image_width = self.image.shape[:2]

#########################
# IMPORT GT ANNOTATIONS #
#########################

    def load_annotation_from_csv(self,
                                 annotations_folder: str,
                                 is_ground_truth: bool = True):
        """
        Load figure annotations from the given (individual) csv file.

        Args:
            annotations_folder (str):   Path to an annotation file (csv format).
            is_ground_truth (bool):     Tells whether annotations are ground truth or predictions.
                                            If True, annotations will be stored in
                                            `self.gt_panels` else, in `self.detected_panels`.
        """
        # Get image file base name (without extension).
        base_name = os.path.splitext(self.image_filename)[0]

        # Compute the path to the annotation csv file.
        annotation_csv = os.path.join(annotations_folder, base_name, '.csv')

        if not os.path.isfile(annotation_csv):
            raise FileNotFoundError("The annotation csv file does not exist :"\
                "\n\tShould be {}".format(annotation_csv))

        # Create empty list of panels
        panels = []

        # Open the csv file containing annotations
        with open(annotation_csv, 'r') as annotation_csv_file:
            csv_reader = csv.reader(annotation_csv_file, delimiter=',')

            # Loop over each row (panel)
            for row in csv_reader:

                # Panel segmentation + panel splitting
                if len(row) == 11:
                    label_coordinates = [int(x) for x in row[6:10]]
                    label = label_class.map_label(row[10])

                # Panel splitting only
                elif len(row) == 6:
                    label_coordinates = None
                    label = None
                else:
                    raise ValueError("Row should be of length 6 or 11.\n\t"\
                                     f"Current row has length {len(row)}: {row}")

                image_path = row[0]
                panel_coordinates = [int(x) for x in row[1:5]]
                panel_class = row[5]
                assert panel_class == 'panel'

                assert image_path == self.image_path, "Wrong image path in csv:"\
                    f"\n\timage file name : {self.image_path}"\
                    f"\n\timage in csv row : {image_path}"

                # Instanciate Panel object
                panel = Panel(panel_rect=panel_coordinates,
                              label_rect=label_coordinates,
                              label=label)

                panels.append(panel)

        # Store the Panel objects in the right class attribute.
        if is_ground_truth:
            self.gt_panels = panels
        else:
            self.detected_panels = panels



    def load_annotation_from_iphotodraw(self,
                                        annotation_file_path: str):
        """
        Load iPhotoDraw annotation.
        Deal with PanelSeg data set.

        Args:
            annotation_file_path (str): The path to the xml file containing annotations.
        """

        def extract_bbox_from_iphotodraw_node(item: ET.Element) -> Tuple[int, int, int, int]:
            """
            Extract bounding box information from Element item (ElementTree).
            It also makes sure that the bounding box is within the image.

            Args:
                item (ET.Element):  Either a panel or label item extracted from an iPhotoDraw
                                        xml annotation file.

            Returns:
                x_min, y_min, x_max, y_max (Tuple[int, int, int, int]): The coordinates of the
                                                                            bounding box
            """
            extent_item = item.find('./Data/Extent')

            # Get data from the xml item.
            height_string = extent_item.get('Height')
            width_string = extent_item.get('Width')

            x_string = extent_item.get('X')
            y_string = extent_item.get('Y')

            # Compute coordinates of the bounding box.
            x_min = round(float(x_string))
            y_min = round(float(y_string))
            x_max = x_min + round(float(width_string))
            y_max = y_min + round(float(height_string))

            # Clip values with respect to the image shape.
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
                panels (List[Panel]): A list of Panel objects.
            """
            panels = []
            for panel_item in panel_items:
                text_item = panel_item.find('./BlockText/Text')
                label_text = text_item.text
                label_text = label_text.strip()
                words = label_text.split(' ')

                # Panels can only have 1 or 2 words:
                # *) The first one is "panel"
                # *) The second one is the label letter
                if len(words) > 2:
                    # TODO check what to do in this case
                    # logging.error(
                        # '%s: %s is not correct',
                        # annotation_file_path,
                        # label_text)
                    continue

                # If the label text contains two words,
                # then the second one is the label text
                if len(words) == 2:
                    label_text = label_class.map_label(words[1])

                    # TODO check what to do in this case
                    # if len(label_text) != 1:
                        # # Now we process single character panel label only (a, b, c...)
                        # logging.warning(
                            # '%s: panel %s is not single character',
                            # annotation_file_path,
                            # label_text)

                # The text only contains a single panel.
                # => no label
                else:
                    label_text = ''

                x_min, y_min, x_max, y_max = extract_bbox_from_iphotodraw_node(item=panel_item)

                if x_max <= x_min or y_max <= y_min:
                    # TODO check what to do in this case
                    # logging.error(
                        # '%s: panel %s rect is not correct!',
                        # annotation_file_path,
                        # label_text)
                    continue

                # Create Panel object
                panel_rect = [x_min, y_min, x_max, y_max]
                panel = Panel(label=label_text,
                              panel_rect=panel_rect,
                              label_rect=None)

                panels.append(panel)

            return panels


        def extract_label_info() -> List[Panel]:
            """
            Extract information from and validate all label items.

            Returns:
                labels (List[Panel]):   A list of Panel objects representing the detected labels.
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
                # TODO check what to do in this case
                # if len(label_text) != 1:
                    # logging.warning(
                        # '%s: label %s is not single character',
                        # annotation_file_path,
                        # label_text)

                label_text = label_class.map_label(label_text)

                x_min, y_min, x_max, y_max = extract_bbox_from_iphotodraw_node(item=label_item)

                if x_max <= x_min or y_max <= y_min:
                    logging.error(f"{annotation_file_path}: label {label_text} rect is not"\
                                   " correct!")
                    continue

                label_rect = [x_min, y_min, x_max, y_max]
                # We use Panel objects temporarily
                label = Panel(label=label_text,
                              panel_rect=None,
                              label_rect=label_rect)

                labels.append(label)

            return labels


        def match_panels_with_labels(panels: List[Panel],
                                     labels: List[Panel]) -> List[Panel]:
            """
            Match both lists to get a unique list of panels containing
            information of their matching label.

            Args:
                panels (List[Panel]):   List of panels without label information.
                labels (List[Panel]):   List of labels without panel information.

            Returns:
                panels (List[Panel]):   Updated list of panels.
            """
            if len(labels) != 0 and len(labels) != len(panels):
                logging.warning(f"{annotation_file_path}: has different panel and label rects."\
                                 " Most likely there are mixes with-label and without-label"\
                                 " panels.")

            # Collect all panel label characters.
            char_set = set()
            for panel in panels:
                if len(panel.label) != 0:
                    char_set.add(panel.label)

            # Build panel dictionary according to labels.
            panel_dict = {s: [] for s in char_set}
            for panel in panels:
                if len(panel.label) != 0:
                    panel_dict[panel.label].append(panel)

            # Build label dictionary according to labels.
            label_dict = {s: [] for s in char_set}
            for label in labels:
                label_dict[label.label].append(label)

            # Assign labels to panels.
            for label_char in char_set:
                if len(panel_dict[label_char]) != len(label_dict[label_char]):
                    logging.error(f"{annotation_file_path}: panel {label_char} does not have"\
                                   " same matching labels!")
                    continue


                # Beam search algorithm to map labels to panels
                # ==> This is only useful when there are several panels with the same label in a
                # single image. In Zou's data set, this is never the case
                beam_search.assign_labels_to_panels(panels=panel_dict[label_char],
                                                    labels=label_dict[label_char])

            # expand the panel_rect to always include label_rect
            for panel in panels:
                if panel.label_rect is not None:
                    panel.panel_rect = box.union(rectangle_1=panel.label_rect,
                                                 rectangle_2=panel.panel_rect)

            return panels


        # Create element tree object.
        tree = ET.parse(annotation_file_path)

        # Get root element.
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
                self._logger.error('%s: has unknown <shape> xml items %s',
                                   annotation_file_path, text)

        # Extract information from and validate all panel items.
        panels = extract_panel_info()

        # Extract information from and validate all label items.
        labels = extract_label_info()

        # Match both lists to get a unique list of panels containing
        # information of their matching label.
        # Save this list of panels in tha appropriate attribute of
        # the figure object.
        self.gt_panels = match_panels_with_labels(panels=panels,
                                                  labels=labels)

##############
# EVALUATION #
##############

    def match_detected_and_gt_panels_splitting_task(self,
                                                    iou_threshold: float = 0.5,
                                                    overlap_threshold: float = 0.66):
        """
        Match the detected panels with a ground truth one.
        The `self.detected_panels` attribute is modified by side effect.
        * A predicted panel which has an IoU > `iou_threshold` (0.5 by default) with a
            ground truth panel is counted as a positive match regarding the "IoU criterion".
            => This criterion is the one to compute precision, recall, mAP.

        * A predicted panel which has an overlap > `overlap_threshold` (0.66 by default) with
            a ground truth panel is counted as a positive match regarding the
            "ImageCLEF citerion".
            => This criterion is the one to compute the ImageCLEF accuracy.
            (see http://ceur-ws.org/Vol-1179/CLEF2013wn-ImageCLEF-SecoDeHerreraEt2013b.pdf)

        ==> Panel splitting task.

        Args:
            iou_threshold (float):      Threshold above which a prediction is considered to be
                                            true with respect to the IoU value.
            overlap_threshold (float):  Threshold above which a prediction is considered to be
                                            true with respect to the overlap value.
        """
        # Keep track of the associations.
        picked_gt_panels_overlap = [False] * len(self.gt_panels)
        picked_gt_panels_iou = [False] * len(self.gt_panels)

        for detected_panel in self.detected_panels:
            max_iou = -1
            max_overlap = -1

            best_matching_gt_panel_iou_index = -1
            best_matching_gt_panel_overlap_index = -1

            for gt_panel_index, gt_panel in enumerate(self.gt_panels):

                intersection_area = box.intersection_area(gt_panel.panel_rect,
                                                          detected_panel.panel_rect)
                if intersection_area == 0:
                    continue

                detected_panel_area = box.area(detected_panel.panel_rect)

                # --> Using ImageCLEF metric (overlap)
                overlap = intersection_area / detected_panel_area

                # Potential match
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_matching_gt_panel_overlap_index = gt_panel_index

                # --> Using IoU (common for object detection)
                iou = box.iou(gt_panel.panel_rect, detected_panel.panel_rect)

                # Potential match
                if iou > max_iou:
                    max_iou = iou
                    best_matching_gt_panel_iou_index = gt_panel_index

            # Check that gt and detected panels are overlapping enough.
            # ==> True positive (w.r.t IoU criterion)
            if max_iou > iou_threshold \
                    and not picked_gt_panels_iou[best_matching_gt_panel_iou_index]:
                picked_gt_panels_iou[best_matching_gt_panel_iou_index] = True
                detected_panel.panel_is_true_positive_iou = True

            # ==> False positive (w.r.t IoU criterion)
            else:
                detected_panel.panel_is_true_positive_iou = False

            # Check that gt and detected panels are overlapping enough.
            # ==> True positive (w.r.t overlap criterion)
            if max_overlap > overlap_threshold \
                    and not picked_gt_panels_overlap[best_matching_gt_panel_overlap_index]:
                picked_gt_panels_overlap[best_matching_gt_panel_overlap_index] = True
                detected_panel.panel_is_true_positive_overlap = True

            # ==> False positive (w.r.t overlap criterion)
            else:
                detected_panel.panel_is_true_positive_overlap = False


    def match_detected_and_gt_labels(self, iou_threshold: float = 0.5):
        """
        Match the detected labels (and their label) with a ground truth one.
        The comparison criterion is the IoU which is maximized.
        The `self.detected_panels` attribute is modified by side effect.

        ==> Label recognition task.

        Args:
            iou_threshold (float):  IoU threshold above which a prediction is considered to be
                                        true.
        """
        # Keep track of the associations.
        picked_gt_panels_indices = [False] * len(self.gt_panels)

        # Loop over detected labels.
        for detected_panel in self.detected_panels:
            max_iou = -1
            best_matching_gt_panel_index = -1

            for gt_panel_index, gt_panel in enumerate(self.gt_panels):

                # TODO laverage the 'single-character label' restriction.
                if gt_panel.label_rect is None or len(gt_panel.label) != 1:
                    continue

                # If the label classes do not match, no need to compute the IoU.
                if gt_panel.label != detected_panel.label:
                    continue

                # Compute IoU between detection and ground truth.
                iou = box.iou(gt_panel.label_rect, detected_panel.label_rect)

                # Potential match
                if iou > max_iou:
                    max_iou = iou
                    best_matching_gt_panel_index = gt_panel_index

            # Check that gt and detected panels are overlapping enough.
            # ==> True positive
            if max_iou > iou_threshold\
                    and not picked_gt_panels_indices[best_matching_gt_panel_index]:
                picked_gt_panels_indices[best_matching_gt_panel_index] = True
                detected_panel.label_is_true_positive = True

            # ==> False positive
            else:
                detected_panel.label_is_true_positive = False


    def match_detected_and_gt_panels_segmentation_task(self,
                                                       iou_threshold: float = 0.5):
        """
        Match the detected panels (and their label) with a ground truth one.
        The comparison criterion is the IoU which is maximized.
        The `self.detected_panels` attribute is modified by side effect.

        ==> Panel segmentation task.

        Args:
            iou_threshold (float):  IoU threshold above which a prediction is considered to be
                                        true.
        """
        # Keep track of the associations.
        picked_gt_panels_indices = [False] * len(self.gt_panels)

        # Loop over detected panels.
        for detected_panel in self.detected_panels:
            max_iou = -1
            best_matching_gt_panel_index = -1

            for gt_panel_index, gt_panel in enumerate(self.gt_panels):

                # TODO laverage the 'single-character label' restriction.
                if gt_panel.label_rect is None or len(gt_panel.label) != 1:
                    continue

                # If the label classes do not match, no need to compute the IoU.
                if gt_panel.label != detected_panel.label:
                    continue

                # Compute IoU between detection and ground truth.
                iou = box.iou(gt_panel.panel_rect, detected_panel.panel_rect)

                # Potential match
                if iou > max_iou:
                    max_iou = iou
                    best_matching_gt_panel_index = gt_panel_index

            # Check that gt and detected panels are overlapping enough.
            # ==> True positive
            if max_iou > iou_threshold\
                    and not picked_gt_panels_indices[best_matching_gt_panel_index]:
                picked_gt_panels_indices[best_matching_gt_panel_index] = True
                detected_panel.panel_is_true_positive = True

            # ==> False positive
            else:
                detected_panel.panel_is_true_positive = False


##################
# PREVIEW FIGURE #
##################

    def get_preview(self, mode='gt') -> np.ndarray:
        """
        Generate an image preview for the figure.
        It consists in drawing the panels (and labels, if applicable) bounding boxes
        on top of the image.

        Args:
            mode: Select which information to display:
                    * 'gt': only the ground truth
                    * 'pred': only the predictions
                    * 'both': both predicted and ground truth annotations.

        Returns:
            preview_img (np.ndarray): the preview image
        """

        if mode not in ('gt', 'pred', 'both'):
            raise ValueError("mode should be either 'gt', 'pred', or 'both'.")

        # Load image if necessary
        if self.image is None:
            self.load_image()
        preview_img = self.image.copy()

        if mode == 'both':

            for panel in self.gt_panels:
                # Green for ground truth panels
                panel.draw_elements(image=preview_img,
                                    color=(0, 255, 0))

            if self.detected_panels is None:
                self._logger.warning("No detected panels exist for this figure." \
                    " Hence, they cannot be displayed.")
                return preview_img

            for panel in self.detected_panels:
                # Yellow for predicted panels
                panel.draw_elements(image=preview_img,
                                    color=(0, 0, 200))

            return preview_img

        if mode == 'gt':
            panels = self.gt_panels

        # mode = 'pred'
        else:
            panels = self.detected_panels

        shape_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]


        if panels is None:
            raise ValueError(f"{mode} panels are None. Cannot display")

        for panel_index, panel in enumerate(panels):

            # Select color
            color = shape_colors[panel_index % len(shape_colors)]

            panel.draw_elements(image=preview_img,
                                color=color)


        # Store the computed image
        self.preview_image = preview_img

        return preview_img


    def show_preview(self,
                     mode: str = 'gt',
                     delay: int = 0,
                     window_name: str = None):
        """
        Display a preview of the image along with the panels and labels drawn on top.

        Args:
            mode (str):         Select which information to display:
                                    * 'gt': only the ground truth
                                    * 'pred': only the predictions
                                    * 'both': both predicted and ground truth annotations.
            delay (int):        The number of seconds after which the window is closed
                                    if 0, the delay is disabled.
            window_name (str):  Name of the image display window.
        """

        image_preview = self.get_preview(mode)

        if window_name is None:
            window_name = self.image_filename

        cv2.imshow(window_name, image_preview)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()


    def save_preview(self,
                     mode: str = 'gt',
                     folder: str = None):
        """
        Save the annotation preview at folder.

        Args:
            mode (str):     Select which information to display:
                                * 'gt': only the ground truth
                                * 'pred': only the predictions
                                * 'both': both predicted and ground truth annotations.
            folder (str):   The folder where to store the image preview.
        """
        # Get the preview image.
        preview_img = self.get_preview(mode)

        # Default output directory.
        if folder is None:
            folder = os.path.join(os.path.dirname(self.image_path),
                                  'preview/')


        # Create the preview directory (if needed).
        if not os.path.isdir(folder):
            os.makedirs(folder)

        # Remove extension from original figure image file name
        file_name = os.path.splitext(
            self.image_filename)[0]

        export_path = os.path.join(folder, file_name + "_preview.jpg")
        export_path = os.path.abspath(export_path)

        # Write the preview image file to destination
        if not cv2.imwrite(export_path, preview_img):
            logging.error("Could not write preview image : %s", export_path)


#################
# EXPORT FIGURE #
#################

    def export_gt_annotation_to_individual_csv(self,
                                               csv_export_dir: str = None):
        """
        Export the ground truth annotation of the figure to an individual csv file.

        Args:
            csv_export_dir (str):   Path to the directory where to export the csv file.
        """

        # By default the csv is at the same location
        if csv_export_dir is None:
            csv_export_dir = os.path.dirname(self.image_path)

        # Check if directory exists.
        if not os.path.isdir(csv_export_dir):
            logging.error(f"Export directory does not exist : {csv_export_dir}")

        # Remove extension from original figure image file name.
        csv_export_file_name = os.path.splitext(self.image_filename)[0] + '.csv'

        # Compute csv annotation file path.
        csv_file_path = os.path.join(csv_export_dir,
                                     csv_export_file_name)

        # Check if file already exists.
        if os.path.isfile(csv_file_path):
            logging.warning(f"The csv individual annotation file already exist : {csv_file_path}"\
                            "\n\t==> Skipping.")
            return

        with open(csv_file_path, 'w', newline='') as csvfile:

            csv_writer = csv.writer(csvfile, delimiter=',')

            # Looping over Panel objects.
            for panel in self.gt_panels:

                # Panel information
                csv_row = [
                    self.image_path,
                    panel.panel_rect[0],
                    panel.panel_rect[1],
                    panel.panel_rect[2],
                    panel.panel_rect[3],
                    'panel'
                    ]

                # Label information
                if panel.label is not None and panel.label_rect is not None:
                    csv_row.append(panel.label_rect[0])
                    csv_row.append(panel.label_rect[1])
                    csv_row.append(panel.label_rect[2])
                    csv_row.append(panel.label_rect[3])
                    csv_row.append(panel.label)

                # Writting to csv file
                csv_writer.writerow(csv_row)


    def convert_to_tf_example(self) -> tf.train.Example:
        """
        Convert the figure (only panel info) to a TensorFlow example which is compatible with the
        TensorFlow Object Detection API.
        This is deprecated since the project relies on Detectron 2 (PyTorch).

        Returns:
            example (tf.train.Example): The corresponding tf example.
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
        for panel in self.gt_panels:
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
