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
from typing import List, Dict
import xml.etree.ElementTree as ET

import numpy as np
import PIL.Image
from cv2 import cv2

from .subfigure import Panel, Label, SubFigure
from . import beam_search
from .. import box
from . import label_class


class Figure:
    """
    A class representing a Figure (and eventually ground truth and/or predicted annotations).

    Attributes:
        index (int):                                    The unique id to each figure.
        image_path (str):                               The path to the figure image file.
        image_filename (str):                           Image file name.
        image_format (str):                             The format of the image file
                                                            (jpg, png,...).
        image (np.ndarray):                             The image data.
        image_width (int):                              The image width (in pixels).
        image_height (int):                             The image height (in pixels).
        caption (str):                                  The complete caption.
        preview_image (np.ndarray):                     The preview image (image + bounding boxes)
        gt_subfigures (List[SubFigure]):                Ground truth subfigure objects.
        detected_panels (List[DetectedPanel]):          Detected panel objects.
        detected_labels (List[DetectedLabel]):          Detected label objects.
        detected_subfigures (List[DetectedSubFigure]):  Detected subfigure objects.
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

        # Caption
        self.caption = None

        # Can contain a preview of the image with its bounding boxes.
        self.preview_image = None

        # Ground truth subfigure annotations.
        self.gt_subfigures = None

        # Detected panels
        self.detected_panels = None

        # Detected labels
        self.detected_labels = None

        # Detected subfigures
        self.detected_subfigures = None

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
                                            `self.gt_subfigures` else, in
                                            `self.detected_subfigures`.
        """
        # Get image file base name (without extension).
        base_name = os.path.splitext(self.image_filename)[0]

        # Compute the path to the annotation csv file.
        annotation_csv = os.path.join(annotations_folder, base_name, '.csv')

        if not os.path.isfile(annotation_csv):
            raise FileNotFoundError("The annotation csv file does not exist :"\
                "\n\tShould be {}".format(annotation_csv))

        # Create empty list of subfigures
        subfigures = []

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
                panel = Panel(box=panel_coordinates)
                label = Label(text=label,
                              box=label_coordinates)

                panel = SubFigure(panel=Panel,
                                  label=Label)

                subfigures.append(panel)

        # Store the Panel objects in the right class attribute.
        if is_ground_truth:
            self.gt_subfigures = subfigures
        else:
            self.detected_subfigures = subfigures



    def load_annotation_from_iphotodraw(self,
                                        annotation_file_path: str):
        """
        Load iPhotoDraw annotation.
        Deal with PanelSeg data set.

        Args:
            annotation_file_path (str): The path to the xml file containing annotations.
        """

        def extract_bbox_from_iphotodraw_node(item: ET.Element) -> box.Box:
            """
            Extract bounding box information from Element item (ElementTree).
            It also makes sure that the bounding box is within the image.

            Args:
                item (ET.Element):  Either a panel or label item extracted from an iPhotoDraw
                                        xml annotation file.

            Returns:
                x_min, y_min, x_max, y_max (box.Box):   The coordinates of the bounding box.
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


        def extract_panel_info() -> Dict[str, Panel]:
            """
            Extract information from and validate all panel items

            Returns:
                panels (Dict[str, Panel]):    A dict linking Panel objects to their labels.
            """
            panels = {}
            for panel_item in panel_items:
                text_item = panel_item.find('./BlockText/Text')
                label_text = text_item.text
                label_text = label_text.strip()
                words = label_text.split(' ')

                # Panels can only have 1 or 2 words:
                # *) The first one is "panel"
                # *) The second one is the label text
                if len(words) > 2:
                    # TODO check what to do in this case
                    self._logger.error(f"{annotation_file_path}: {label_text} is not correct")
                    continue

                # If the label text contains two words,
                # then the second one is the label text
                elif len(words) == 2:
                    label_text = label_class.map_label(words[1])

                    # TODO check what to do in this case
                    if len(label_text) != 1:
                        # # Now we process single character panel label only (a, b, c...)
                        self._logger.warning(f"{annotation_file_path}: panel {label_text}"\
                                              " is not single character")

                # The text only contains a single panel.
                # => no label
                else:
                    label_text = ''

                x_min, y_min, x_max, y_max = extract_bbox_from_iphotodraw_node(item=panel_item)

                if x_max <= x_min or y_max <= y_min:
                    # TODO check what to do in this case
                    self._logger.error(f"{annotation_file_path}: panel {label_text} rect is not"\
                                   " correct!")
                    continue

                # Create Panel object
                panel_rect = [x_min, y_min, x_max, y_max]
                panel = Panel(box=panel_rect)

                if label_text in panels:
                    panels[label_text].append(panel)
                else:
                    panels[label_text] = [panel]

            return panels


        def extract_label_info() -> List[Label]:
            """
            Extract information from and validate all label items.

            Returns:
                labels (List[Label]):   A list of Label objects representing the detected labels.
            """
            labels = []
            for label_item in label_items:
                text_item = label_item.find('./BlockText/Text')
                label_text = text_item.text
                label_text = label_text.strip()
                words = label_text.split(' ')

                # Labels can only have 1 or 2 words:
                # *) The first one is "label"
                # *) The second one is the label text
                if len(words) != 2:
                    self._logger.error(f"{annotation_file_path}: {label_text} is not correct")
                    continue
                label_text = words[1]
                # Now we process single character panel label only
                # TODO check what to do in this case
                if len(label_text) != 1:
                    self._logger.warning(f"{annotation_file_path}: label {label_text} is not"\
                                          " single character")

                label_text = label_class.map_label(label_text)

                x_min, y_min, x_max, y_max = extract_bbox_from_iphotodraw_node(item=label_item)

                if x_max <= x_min or y_max <= y_min:
                    self._logger.error(f"{annotation_file_path}: label {label_text} rect is not"\
                                        " correct!")
                    continue

                label_rect = [x_min, y_min, x_max, y_max]

                # Instanciate Label object.
                label = Label(text=label_text,
                              box=label_rect)

                labels.append(label)

            return labels


        def match_panels_with_labels(panels: Dict[str, Panel],
                                     labels: List[Label]
                                     ) -> List[SubFigure]:
            """
            Match both lists to get a unique list of panels containing
            information of their matching label.

            Args:
                panels (Dict[str, Panel]):  Dict linking panels and their label.
                labels (List[Label]):       List of labels.

            Returns:
                subfigures (List[SubFigure]):   List of subfigures (containing panel and label
                                                    information).
            """
            if len(labels) != 0 and len(labels) != len(panels):
                self._logger.warning(f"{annotation_file_path}: has different panel and label"\
                                      " rects. Most likely there are mixes with-label and"\
                                      " without-label panels.")

            # Collect all panel label characters.
            char_set = set()
            for panel_label in panels.keys():
                if len(panel_label) != 0:
                    char_set.add(panel_label)

            # Build panel dictionary according to labels.
            panel_dict = {s: [] for s in char_set}
            for panel_label, panel in panels.items():
                if len(panel_label) != 0:
                    panel_dict[panel_label].append(panel)

            # Build label dictionary according to labels.
            label_dict = {s: [] for s in char_set}
            for label in labels:
                label_dict[label.text].append(label)

            subfigures = []
            # Assign labels to panels.
            for label_char in char_set:
                if len(panel_dict[label_char]) != len(label_dict[label_char]):
                    self._logger.error(f"{annotation_file_path}: panel {label_char} does not"\
                                        " have same matching labels!")
                    continue

                # if len(panel_dict[label_char]) > 1:
                    # print(panel_dict)

                panel = panel_dict[label_char][0]

            # Expand the panel_rect to always include label_rect.
            for subfigure in subfigures:
                panel = subfigure.panel
                label = subfigure.label
                if label.box is not None:
                    panel.box = box.union(box_1=panel.box,
                                          box_2=label.box)

            return subfigures


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
                self._logger.error("{annotation_file_path}: has unknown <shape> xml items {text}")

        # Extract information from and validate all panel items.
        panels = extract_panel_info()
        # if len(panel_items) != len(panels):
            # print(f"len(panel_items) = {len(panel_items)}")
            # print(f"panel_items = {panel_items}")
            # print(f"panels = {panels}")

        # Extract information from and validate all label items.
        labels = extract_label_info()
        # if len(label_items) != len(labels):
            # # print(f"label_items = {label_items}")
            # print(f"labels = {labels}")

        # Match both lists to get a unique list of panels containing
        # information of their matching label.
        # Save this list of panels in tha appropriate attribute of
        # the figure object.
        self.gt_subfigures = match_panels_with_labels(panels=panels,
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
        picked_gt_panels_overlap = [False] * len(self.gt_subfigures)
        picked_gt_panels_iou = [False] * len(self.gt_subfigures)

        for detected_panel in self.detected_panels:
            max_iou = -1
            max_overlap = -1

            best_matching_gt_panel_iou_index = -1
            best_matching_gt_panel_overlap_index = -1

            for gt_panel_index, gt_subfigure in enumerate(self.gt_subfigures):

                gt_panel = gt_subfigure.panel

                intersection_area = box.intersection_area(gt_panel.box,
                                                          detected_panel.box)
                if intersection_area == 0:
                    continue

                detected_panel_area = box.area(detected_panel.box)

                # --> Using ImageCLEF metric (overlap)
                overlap = intersection_area / detected_panel_area

                # Potential match
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_matching_gt_panel_overlap_index = gt_panel_index

                # --> Using IoU (common for object detection)
                iou = box.iou(gt_panel.box, detected_panel.box)

                # Potential match
                if iou > max_iou:
                    max_iou = iou
                    best_matching_gt_panel_iou_index = gt_panel_index

            # Check that gt and detected panels are overlapping enough.
            # ==> True positive (w.r.t IoU criterion)
            if max_iou > iou_threshold \
                    and not picked_gt_panels_iou[best_matching_gt_panel_iou_index]:
                picked_gt_panels_iou[best_matching_gt_panel_iou_index] = True
                detected_panel.is_true_positive_iou = True

            # ==> False positive (w.r.t IoU criterion)
            else:
                detected_panel.is_true_positive_iou = False

            # Check that gt and detected panels are overlapping enough.
            # ==> True positive (w.r.t overlap criterion)
            if max_overlap > overlap_threshold \
                    and not picked_gt_panels_overlap[best_matching_gt_panel_overlap_index]:
                picked_gt_panels_overlap[best_matching_gt_panel_overlap_index] = True
                detected_panel.is_true_positive_overlap = True

            # ==> False positive (w.r.t overlap criterion)
            else:
                detected_panel.is_true_positive_overlap = False


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
        picked_gt_labels_indices = [False] * len(self.gt_subfigures)

        # Loop over detected labels.
        for detected_label in self.detected_labels:
            max_iou = -1
            best_matching_gt_label_index = -1

            for gt_label_index, gt_subfigure in enumerate(self.gt_subfigures):

                gt_label = gt_subfigure.label

                # TODO laverage the 'single-character label' restriction.
                if gt_label.box is None or len(gt_label.text) != 1:
                    continue

                # If the label classes do not match, no need to compute the IoU.
                if gt_label.text != detected_label.text:
                    continue

                # Compute IoU between detection and ground truth.
                iou = box.iou(gt_label.box, detected_label.box)

                # Potential match
                if iou > max_iou:
                    max_iou = iou
                    best_matching_gt_label_index = gt_label_index

            # Check that gt and detected labels are overlapping enough.
            # ==> True positive
            if max_iou > iou_threshold\
                    and not picked_gt_labels_indices[best_matching_gt_label_index]:
                picked_gt_labels_indices[best_matching_gt_label_index] = True
                detected_label.is_true_positive = True

            # ==> False positive
            else:
                detected_label.is_true_positive = False


    def match_detected_and_gt_panels_segmentation_task(self,
                                                       iou_threshold: float = 0.5):
        """
        Match the detected subfigures with a ground truth one.
        The comparison criterion is the IoU between panels which is maximized.
        The `self.detected_subfigures` attribute is modified by side effect.

        ==> Panel segmentation task.

        Args:
            iou_threshold (float):  IoU threshold above which a prediction is considered to be
                                        true.
        """
        # Keep track of the associations.
        picked_gt_subfigures_indices = [False] * len(self.gt_subfigures)

        # Loop over detected subfigures.
        for detected_subfigure in self.detected_subfigures:
            max_iou = -1
            best_matching_gt_subfigure_index = -1

            detected_panel = detected_subfigure.panel
            detected_label = detected_subfigure.label

            for gt_subfigure_index, gt_subfigure in enumerate(self.gt_subfigures):

                gt_panel = gt_subfigure.panel
                gt_label = gt_subfigure.label

                # TODO laverage the 'single-character label' restriction.
                if gt_label.box is None or len(gt_label.text) != 1:
                    continue

                # If the label classes do not match, no need to compute the IoU.
                if gt_label.text != detected_label.text:
                    continue

                # Compute IoU between detection and ground truth.
                iou = box.iou(gt_panel.box, detected_panel.box)

                # Potential match
                if iou > max_iou:
                    max_iou = iou
                    best_matching_gt_subfigure_index = gt_subfigure_index

            # Check that gt and detected panels are overlapping enough.
            # ==> True positive
            if max_iou > iou_threshold\
                    and not picked_gt_subfigures_indices[best_matching_gt_subfigure_index]:
                picked_gt_subfigures_indices[best_matching_gt_subfigure_index] = True
                detected_subfigure.is_true_positive = True

            # ==> False positive
            else:
                detected_subfigure.is_true_positive = False


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
        # Load image if necessary
        if self.image is None:
            self.load_image()
        preview_img = self.image.copy()


        # Prepare set of colors for drawing elements:
        shape_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        # 1st case ('both'): Display both gt and detections.
        if mode == 'both':

            # Display ground truth subfigures.
            for subfigure in self.gt_subfigures:
                subfigure.draw_elements(image=preview_img)

            # If this figure contains detected subfigures, they are considered to be
            # the relevant detections to display.
            if self.detected_subfigures is not None:

                for detected_subfigure in self.detected_subfigures:
                    detected_subfigure.draw_elements(image=preview_img)

            # Else, we display the available detected elements.
            else:
                if self.detected_panels is not None:
                    for panel in self.detected_panels:
                        panel.draw(image=preview_img)

                if self.detected_labels is not None:
                    for label in self.detected_labels:
                        label.draw(image=preview_img)


        # 2nd case ('gt'): Display ground-truth subfigures.
        elif mode == 'gt':

            # Display ground truth subfigures.
            for subfigure_index, subfigure in enumerate(self.gt_subfigures):

                # Select color.
                color = shape_colors[subfigure_index % len(shape_colors)]

                # Draw subfigure.
                subfigure.draw_elements(image=preview_img,
                                        color=color)


        # 3rd case ('pred'): Display detections.
        elif mode == 'pred':

            # If this figure contains detected subfigures, they are considered to be
            # the relevant detections to display.
            if self.detected_subfigures is not None:

                for subfigure_index, detected_subfigure in enumerate(self.detected_subfigures):
                    # Select color.
                    color = shape_colors[subfigure_index % len(shape_colors)]

                    # Draw element.
                    detected_subfigure.draw_elements(image=preview_img,
                                                     color=color)

            # Else, we display the available detected elements.
            else:
                if self.detected_panels is not None:
                    for panel_index, panel in enumerate(self.detected_panels):
                        # Select color.
                        color = shape_colors[panel_index % len(shape_colors)]

                        # Draw element.
                        panel.draw(image=preview_img,
                                   color=color)

                if self.detected_labels is not None:
                    for label_index, label in enumerate(self.detected_labels):
                        # Select color.
                        color = shape_colors[label_index % len(shape_colors)]

                        # Draw element.
                        label.draw(image=preview_img,
                                   color=color)

        # Invalid mode.
        else:
            raise ValueError("`mode` should be either 'gt', 'pred', or 'both'.")

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
            for subfigure in self.gt_subfigures:

                # Panel information
                csv_row = [
                    self.image_path,
                    subfigure.panel.box[0],
                    subfigure.panel.box[1],
                    subfigure.panel.box[2],
                    subfigure.panel.box[3],
                    'panel'
                    ]

                # Label information
                if subfigure.label is not None:
                    # Label bounding box
                    csv_row.append(subfigure.label.box[0])
                    csv_row.append(subfigure.label.box[1])
                    csv_row.append(subfigure.label.box[2])
                    csv_row.append(subfigure.label.box[3])

                    # Label text
                    csv_row.append(subfigure.label.text)

                # Writting to csv file.
                csv_writer.writerow(csv_row)


    def convert_to_tf_example(self) -> 'tf.train.Example':
        """
        Convert the figure (only panel info) to a TensorFlow example which is compatible with the
        TensorFlow Object Detection API.
        This is deprecated since the project relies on Detectron 2 (PyTorch).

        Returns:
            example (tf.train.Example): The corresponding tf example.
        """
        # Import TensorFlow.
        import tensorflow as tf
        from .. import dataset_util

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
        for subfigure in self.gt_subfigures:

            panel = subfigure.panel

            # Bounding box
            xmin.append(float(panel.box[0]) / self.image_width)
            ymin.append(float(panel.box[1]) / self.image_height)
            xmax.append(float(panel.box[2]) / self.image_width)
            ymax.append(float(panel.box[3]) / self.image_height)

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
