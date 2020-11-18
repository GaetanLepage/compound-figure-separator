"""
#############################
#        CompFigSep         #
# Compound Figure Separator #
#############################

GitHub:         https://github.com/GaetanLepage/compound-figure-separator

Author:         Gaétan Lepage
Email:          gaetan.lepage@grenoble-inp.fr
Date:           Spring 2020

Master's project @HES-SO (Sierre, SW)

Supervisors:    Henning Müller (henning.mueller@hevs.ch)
                Manfredo Atzori (manfredo.atzori@hevs.ch)

Collaborators:  Niccolò Marini (niccolo.marini@hevs.ch)
                Stefano Marchesin (stefano.marchesin@unipd.it)


############################
Class representing a figure.
"""

from __future__ import annotations

import os
import csv
import logging
from typing import cast, List, Dict, Set
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np # type: ignore
from cv2 import cv2 # type: ignore

from .sub_figure import SubFigure, DetectedSubFigure, Color
from .panel import Panel, DetectedPanel
from .label import (Label,
                    DetectedLabel,
                    LabelStructure,
                    label_class)

from . import beam_search
from .. import box, Box


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
        labels_structure (LabelStructure):              Object defining entirely the labels of the
                                                            figure.
        caption (str):                                  The complete caption.
        detected_subcaptions (Dict[str, str]):          The dict containing detected subcaptions.
        preview_image (np.ndarray):                     The preview image (image + bounding boxes)
        gt_subfigures (List[SubFigure]):                Ground truth subfigure objects.
        detected_panels (List[DetectedPanel]):          Detected panel objects.
        detected_labels (List[DetectedLabel]):          Detected label objects.
        detected_subfigures (List[DetectedSubFigure]):  Detected subfigure objects.
    """

    def __init__(self,
                 image_path: str,
                 index: int) -> None:
        """
        Init for a Figure object.  Neither the image or the annotations are loaded at this stage.

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

        # Structure of the labels
        self.labels_structure: LabelStructure

        # Caption
        self.caption: str
        # Dicts mapping detected sub-captions to labels.
        self.gt_subcaptions: OrderedDict
        self.detected_subcaptions: OrderedDict

        # Can contain a preview of the image with its bounding boxes.
        self.preview_image: np.ndarray

        # Ground truth subfigure annotations.
        self.gt_subfigures: List[SubFigure]

        # Detected panels
        self.detected_panels: List[DetectedPanel]

        # Detected labels
        self.detected_labels: List[DetectedLabel]

        # Detected subfigures
        self.detected_subfigures: List[DetectedSubFigure]

        # Logger
        self._logger = logging.getLogger(__name__)


    @classmethod
    def from_dict(cls,
                  figure_dict: Dict,
                  index: int) -> Figure:
        """
        Create a Figure object from a dictionnary.

        Args:
            figure_dict (Dict): A dictionnary representing the figure information.

        Returns:
            figure (Figure):    The resulting Figure object.
        """
        figure = Figure(image_path=figure_dict['image_path'],
                        index=index)

        figure.image_width = figure_dict['image_width']
        figure.image_height = figure_dict['image_height']

        if 'gt_subfigures' in figure_dict:
            figure.gt_subfigures = [SubFigure.from_dict(subfigure_dict)
                                    for subfigure_dict in figure_dict['gt_subfigures']]

        if 'detected_subfigures' in figure_dict:
            figure.detected_subfigures = [DetectedSubFigure.from_dict(detected_subfigure_dict)
                                          for detected_subfigure_dict
                                          in figure_dict['detected_subfigures']]

        if 'caption' in figure_dict:
            figure.caption = figure_dict['caption']

        if 'detected_panels' in figure_dict:
            figure.detected_panels = [DetectedPanel.from_dict(panel_dict)
                                      for panel_dict in figure_dict['detected_panels']]

        if 'detected_labels' in figure_dict:
            figure.detected_labels = [DetectedLabel.from_dict(label_dict)
                                      for label_dict in figure_dict['detected_labels']]

        if 'detected_subcaptions' in figure_dict:
            figure.detected_subcaptions = figure_dict['detected_subcaptions']


        return figure


    def load_image(self) -> None:
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
        self.image = img

        # Store the image size
        self.image_height, self.image_width = img.shape[:2]


#########################
# IMPORT GT ANNOTATIONS #
#########################

    def load_annotation_from_csv(self,
                                 annotations_folder: str,
                                 is_ground_truth: bool = True) -> None:
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
        subfigures: List[SubFigure] = []

        # Open the csv file containing annotations
        with open(annotation_csv, 'r') as annotation_csv_file:
            csv_reader = csv.reader(annotation_csv_file, delimiter=',')

            # Loop over each row (panel)
            for row in csv_reader:

                # Panel segmentation + panel splitting
                if len(row) == 11:
                    label_coordinates: Box = cast(Box,
                                                  tuple(int(x)
                                                        for x in row[6:10]))

                    label_text: str = label_class.map_label(row[10])

                    label: Label = Label(text=label_text,
                                         box=label_coordinates)

                # Panel splitting only
                elif len(row) != 6:
                    raise ValueError("Row should be of length 6 or 11.\n\t"\
                                     f"Current row has length {len(row)}: {row}")

                image_path: str = row[0]
                panel_coordinates: Box = cast(Box,
                                              tuple(int(x) for x in row[1:5]))
                panel_class: str = row[5]
                assert panel_class == 'panel'

                assert image_path == self.image_path,\
                    "Wrong image path in csv:"\
                    f"\n\timage file name : {self.image_path}"\
                    f"\n\timage in csv row : {image_path}"

                # Instanciate Panel object
                panel: Panel = Panel(box=panel_coordinates)

                subfigure: SubFigure = SubFigure(panel=panel,
                                                 label=label)

                subfigures.append(subfigure)

        # Store the Panel objects in the right class attribute.
        if is_ground_truth:
            self.gt_subfigures = subfigures
        else:
            self.detected_subfigures = [DetectedSubFigure.from_normal_sub_figure(subfigure)
                                        for subfigure in subfigures]


    def load_annotation_from_iphotodraw(self,
                                        annotation_file_path: str) -> None:
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


        def extract_panel_info() -> Dict[str, List[Panel]]:
            """
            Extract information from and validate all panel items

            Returns:
                panel_dict (Dict[str, Panel]):  A dict linking Panel objects to their labels.
            """
            panel_dict: Dict[str, List[Panel]] = {}

            for panel_item in panel_items:
                text_item = panel_item.find('./BlockText/Text')
                label_text: str = text_item.text
                label_text = label_text.strip()
                words: str = label_text.split(' ')

                # Panels can only have 1 or 2 words:
                # *) The first one is "panel"
                # *) The second one is the label text
                if len(words) > 2:
                    # The panel annotation is not valid. Skip it.
                    self._logger.error("%s: %s is not correct",
                                       annotation_file_path,
                                       label_text)
                    continue

                # If the label text contains two words,
                # then the second one is the label text
                if len(words) == 2:
                    label_text = words[1]

                # The text only contains a single panel.
                # => no label
                else:
                    label_text = ''

                x_min, y_min, x_max, y_max = extract_bbox_from_iphotodraw_node(item=panel_item)

                if x_max <= x_min or y_max <= y_min:
                    # TODO check what to do in this case
                    self._logger.error("%s: panel %s rect is not correct!",
                                       annotation_file_path,
                                       label_text)
                    continue

                # Create Panel object
                panel_rect: box.Box = (x_min, y_min, x_max, y_max)
                panel: Panel = Panel(box=panel_rect)

                if label_text in panel_dict:
                    panel_dict[label_text].append(panel)
                else:
                    panel_dict[label_text] = [panel]

            return panel_dict


        def extract_label_info() -> Dict[str, List[Label]]:
            """
            Extract information from and validate all label items.

            Returns:
                label_dict (Dict[str, List[Label]]):    A list of Label objects representing the
                                                            detected labels.
            """
            label_dict: Dict[str, List[Label]] = {}

            for label_item in label_items:
                text_item = label_item.find('./BlockText/Text')
                label_text: str = text_item.text
                label_text = label_text.strip()
                words: str = label_text.split(' ')

                # Labels can only have 2 words:
                # *) The first one is "label"
                # *) The second one is the label text
                if len(words) != 2:
                    self._logger.error("%s: %s is not correct",
                                       annotation_file_path,
                                       label_text)
                    continue
                label_text = words[1]

                x_min, y_min, x_max, y_max = extract_bbox_from_iphotodraw_node(item=label_item)

                if x_max <= x_min or y_max <= y_min:
                    self._logger.error("%s: label %s rect is not correct!",
                                       annotation_file_path,
                                       label_text)
                    continue

                label_rect = (x_min, y_min, x_max, y_max)

                # Instanciate Label object.
                label = Label(text=label_text,
                              box=label_rect)

                if label_text in label_dict:
                    label_dict[label_text].append(label)
                else:
                    label_dict[label_text] = [label]

            return label_dict


        def match_panels_with_labels(panel_dict: Dict[str, List[Panel]],
                                     label_dict: Dict[str, List[Label]]) -> List[SubFigure]:
            """
            Match both lists to get a unique list of subfigures.

            Args:
                panel_dict (Dict[str, List[Panel]]):    Dict linking Panel objects to the
                                                            associated label text.
                label_dict (Dict[str, List[Panel]]):    Dict linking Label objects to their
                                                            associated label text.

            Returns:
                subfigures (List[SubFigure]):   List of subfigures (containing panel and label
                                                    information).
            """
            # Resulting list of subfigures.
            subfigures: List[SubFigure] = []

            # First, extract unlabeled panels.
            if '' in panel_dict:
                unlabeled_panels = panel_dict.pop('')
                subfigures.extend(SubFigure(panel=panel)
                                  for panel in unlabeled_panels)

            # Case where the figure has no labels.
            if len(label_dict) == 0:
                if len(panel_dict) != 0:
                    self._logger.error("%s: Some panels have label annotations (%s) but there"\
                                       " are no labels.",
                                       annotation_file_path,
                                       panel_dict)

                return subfigures

            # Case where the figure contains labels.
            # => Check that there are as many labeled panels as there are labels.
            num_labeled_panels: int = sum(len(panels)
                                          for panels in panel_dict.values())
            num_labels: int = sum(len(labels)
                                  for labels in label_dict.values())

            if num_labeled_panels != num_labels:
                self._logger.error("%s has a different number of labeled"\
                                   " panels and labels:\n%s\n%s",
                                   annotation_file_path,
                                   str(panel_dict),
                                   str(label_dict))
                return subfigures

            # Collect all panel label characters.
            label_texts: Set = set(panel_dict.keys())

            # Assign labels to panels.
            for label_text in label_texts:
                # Check if, for the same label, the number of panels and labels is the same
                # (should be 1).
                if len(panel_dict[label_text]) != len(label_dict[label_text]):

                    self._logger.error("%s: For label %s, there is not the same number of panels"\
                                       " (%s) and labels (%s) have same matching labels!",
                                       annotation_file_path,
                                       label_text,
                                       panel_dict[label_text],
                                       label_dict[label_text])
                    continue

                # Multiple panel/label pairs for the same label text.
                if len(panel_dict[label_text]) > 1:
                    self._logger.info("%s: Multiple panels/labels with same label %s."\
                                      " Matching them with beam search.",
                                      annotation_file_path,
                                      label_text)

                    subfigures.extend(beam_search.assign_labels_to_panels(
                        panels=panel_dict[label_text],
                        labels=label_dict[label_text]))

                # Single panel and label for the same label text.
                else:
                    panel: Panel = panel_dict[label_text][0]
                    label: Label = label_dict[label_text][0]

                    subfigures.append(SubFigure(panel=panel,
                                                label=label))

            # Expand the panel_rect to always include label_rect.
            for subfigure in subfigures:

                if subfigure.label is not None \
                    and subfigure.label.box is not None\
                    and subfigure.panel is not None:

                    subfigure.panel.box = box.union(box_1=subfigure.panel.box,
                                                    box_2=subfigure.label.box)

            return subfigures


        # Create element tree object.
        tree: ET.ElementTree = ET.parse(annotation_file_path)

        # Get root element.
        root: ET.Element = tree.getroot()

        shape_items: List[ET.Element] = root.findall('./Layers/Layer/Shapes/Shape')

        # Read All Items (Panels and Labels)
        panel_items: List[Panel] = []
        label_items: List[Label] = []
        for shape_item in shape_items:
            text_item: ET.Element = shape_item.find('./BlockText/Text')
            text: str = text_item.text.lower()
            if text.startswith('panel'):
                panel_items.append(shape_item)
            elif text.startswith('label'):
                label_items.append(shape_item)
            else:
                self._logger.error("{annotation_file_path}: has unknown <shape> xml items {text}")

        # Extract information from and validate all panel items.
        panel_dict = extract_panel_info()

        # Extract information from and validate all label items.
        label_dict = extract_label_info()

        # Match both lists to get a unique list of panels containing
        # information of their matching label.
        # Save this list of panels in tha appropriate attribute of
        # the figure object.
        self.gt_subfigures = match_panels_with_labels(panel_dict=panel_dict,
                                                      label_dict=label_dict)


    def load_caption_annotation(self) -> None:
        """
        Load caption text and the ground truth for sub captions (if available).
        """
        caption_annotation_file_path = self.image_path.replace('.jpg', '_caption.txt')

        if not os.path.isfile(caption_annotation_file_path):
            logging.info("Caption file missing:\nNo such file as %s.",
                         caption_annotation_file_path)
            return

        with open(caption_annotation_file_path, 'r') as caption_annotation_file:
            lines = caption_annotation_file.readlines()

        if len(lines) == 0:
            logging.warning("Caption annotation file %s seems to be empty.",
                            caption_annotation_file_path)
            return

        self.caption = lines[0]

        if len(lines) <= 1:
            return

        labels_line = lines[1]
        labels_list = [label.strip() for label in labels_line.split(',')]
        labels_structure: LabelStructure = LabelStructure.from_labels_list(labels_list)

        if hasattr(self, 'labels_structure'):
            assert labels_structure == self.labels_structure,\
                "LabelStructure do not correspond:"\
                f"\n{labels_structure} != {self.labels_structure}"

        else:
            self.labels_structure = labels_structure

        caption_dict: Dict[str, str] = {}
        for caption_line in lines[2:]:
            label, text = caption_line.split(':', maxsplit=1)
            label = label.strip()
            text = text.strip()
            caption_dict[label] = text

        # Store the captions gt.
        self.gt_subcaptions = OrderedDict(dict(sorted(caption_dict.items(),
                                                      key=lambda item: item[0])))

        # Loop over gt_subfigures.
        for gt_subfigure in self.gt_subfigures:

            # If it has a label,
            if gt_subfigure.label is not None:
                label_object = gt_subfigure.label

                # and if that label has a valid label text,
                if label_object.text is not None and len(label_object.text) == 1:

                    # Look for a matching caption.
                    for caption_label in caption_dict:

                        # If the labels "match":
                        if caption_label in label_object.text:

                            # Augment the Label object with the caption text.
                            gt_subfigure.caption = caption_dict.pop(caption_label)
                            break


##############
# EVALUATION #
##############

    def match_detected_and_gt_panels_splitting_task(self,
                                                    iou_threshold: float = 0.5,
                                                    overlap_threshold: float = 0.66) -> None:
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
            max_iou: float = -1
            max_overlap: float = -1

            best_matching_gt_panel_iou_index: int = -1
            best_matching_gt_panel_overlap_index: int = -1

            for gt_panel_index, gt_subfigure in enumerate(self.gt_subfigures):

                gt_panel = gt_subfigure.panel

                if gt_panel is None:
                    continue

                intersection_area: float = box.intersection_area(gt_panel.box,
                                                                 detected_panel.box)
                if intersection_area == 0:
                    continue

                detected_panel_area: float = box.area(detected_panel.box)

                # --> Using ImageCLEF metric (overlap)
                overlap: float = intersection_area / detected_panel_area

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


    def match_detected_and_gt_labels(self,
                                     iou_threshold: float = 0.5) -> None:
        """
        Match the detected labels with a ground truth one.
        The comparison criterion is the IoU which is maximized.
        The `self.detected_labels` attribute is modified by side effect.

        ==> Label recognition task.

        Args:
            iou_threshold (float):  IoU threshold above which a prediction is considered to be
                                        true.
        """
        # Keep track of the associations.
        picked_gt_labels_indices = [False] * len(self.gt_subfigures)

        # Loop over detected labels.
        for detected_label in self.detected_labels:
            max_iou: float = -1
            best_matching_gt_label_index: int = -1

            for gt_label_index, gt_subfigure in enumerate(self.gt_subfigures):

                if gt_subfigure.label is None:
                    continue

                gt_label: Label = gt_subfigure.label

                # TODO laverage the 'single-character label' restriction.
                if gt_label.box is None\
                    or gt_label.text is None\
                    or len(gt_label.text) != 1:

                    continue

                # If the label classes do not match, no need to compute the IoU.
                if gt_label.text != detected_label.text:
                    continue

                assert detected_label.box is not None

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
                                                       iou_threshold: float = 0.5) -> None:
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


    def match_detected_and_gt_captions(self) -> None:
        """
        Match the detected captions with a ground truth one.
        The comparison criterion is the IoU (adapted to text) which is maximized.
        The `self.detected_captions` attribute is modified by side effect.

        TODO: better match panels and labels and then, evaluate caption splitting.

        ==> Caption splitting task.
        """
        # Keep track of the associations.
        picked_gt_captions_indices = [False] * len(self.gt_subfigures)

        # Loop over detected labels.
        for detected_caption in self.detected_captions:
            max_iou = -1
            best_matching_gt_caption_index = -1

            for gt_caption_index, gt_subfigure in enumerate(self.gt_subfigures):

                gt_caption = gt_subfigure.caption

                if gt_caption is None or gt_caption == "":
                    continue

                # Compute IoU between detection and ground truth.
                iou = text.iou(gt_caption, detected_caption)

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


##################
# PREVIEW FIGURE #
##################

    def get_preview(self, mode='both') -> np.ndarray:
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

        assert self.image is not None

        preview_img: np.ndarray = self.image.copy()


        # Prepare set of colors for drawing elements:
        shape_colors: List[Color] = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        # Automatically set mode to 'gt' if there is no detected elements.
        if mode == 'both' and not hasattr(self, 'detected_subfigures') \
                          and not hasattr(self, 'detected_panels') \
                          and not hasattr(self, 'detected_labels'):

            mode = 'gt'

        # 1st case ('both'): Display both gt and detections.
        if mode == 'both':

            # Display ground truth subfigures.
            for subfigure in self.gt_subfigures:
                subfigure.draw_elements(image=preview_img)

            # If this figure contains detected subfigures, they are considered to be
            # the relevant detections to display.
            if hasattr(self, 'detected_subfigure'):

                for detected_subfigure in self.detected_subfigures:
                    detected_subfigure.draw_elements(image=preview_img)

            # Else, we display the available detected elements.
            else:
                if hasattr(self, 'detected_panels'):
                    for panel in self.detected_panels:
                        panel.draw(image=preview_img)

                if hasattr(self, 'detected_labels'):
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
                     window_name: str = None) -> None:
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
                     folder: str = None) -> None:
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
                                               csv_export_dir: str = None) -> None:
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
            logging.error("Export directory does not exist : %s",
                          csv_export_dir)

        # Remove extension from original figure image file name.
        csv_export_file_name = os.path.splitext(self.image_filename)[0] + '.csv'

        # Compute csv annotation file path.
        csv_file_path = os.path.join(csv_export_dir,
                                     csv_export_file_name)

        # Check if file already exists.
        if os.path.isfile(csv_file_path):
            logging.warning("The csv individual annotation file already exist : %s"\
                            "\n\t==> Skipping.", csv_file_path)
            return

        # Open annotation file
        with open(csv_file_path, 'w', newline='') as csvfile:

            csv_writer = csv.writer(csvfile, delimiter=',')

            # Looping over Panel objects.
            for subfigure in self.gt_subfigures:

                if subfigure.panel is None:
                    continue

                # Panel information
                csv_row = [self.image_path,
                           subfigure.panel.box[0],
                           subfigure.panel.box[1],
                           subfigure.panel.box[2],
                           subfigure.panel.box[3],
                           'panel']

                # Label information
                if subfigure.label is not None \
                    and subfigure.label.box is not None:

                    # Label bounding box
                    csv_row.append(subfigure.label.box[0])
                    csv_row.append(subfigure.label.box[1])
                    csv_row.append(subfigure.label.box[2])
                    csv_row.append(subfigure.label.box[3])

                    # Label text
                    csv_row.append(subfigure.label.text)

                # Writting to csv file.
                csv_writer.writerow(csv_row)


    def to_dict(self) -> Dict:
        """
        Export to a dict.

        Returns:
            output_dict (Dict): A Dict representing the figure information.
        """

        output_dict = {
            'image_path': self.image_path,
            'image_width': self.image_width,
            'image_height': self.image_height
        }

        if hasattr(self, 'gt_subfigures'):
            output_dict['gt_subfigures'] = [subfigure.to_dict()
                                            for subfigure in self.gt_subfigures]

        if hasattr(self, 'detected_subfigures'):
            output_dict['detected_subfigures'] = [subfigure.to_dict()
                                                  for subfigure in self.detected_subfigures]

        if hasattr(self, 'caption'):
            output_dict['caption'] = self.caption

        if hasattr(self, 'detected_panels'):
            output_dict['detected_panels'] = [panel.to_dict()
                                              for panel in self.detected_panels]

        if hasattr(self, 'detected_labels'):
            output_dict['detected_labels'] = [label.to_dict()
                                              for label in self.detected_labels]

        if hasattr(self, 'detected_subcaptions'):
            output_dict['detected_subcaptions'] = self.detected_subcaptions

        return output_dict
