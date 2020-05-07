"""
Class representing a figure.
"""

import os
import csv
import logging
import io
import hashlib
from typing import List
import xml.etree.ElementTree as ET

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
    A class for a Figure

    Attributes:
        image_path: is the path to the figure image file
        id: is the unique id to each figure
        image_orig: is the original color image
        panels: contain all panels
    """

    def __init__(self, image_path: str, index: int):
        """
        Constructor for a Figure object.

        Args:
            image_path: path to the image file
            index:      A unique index to identify the figure within the data set
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
        self.pred_panels = None

        # Logger
        self._logger = logging.getLogger(__name__)


    def load_image(self):
        """
        Load the image using `self.image_path` and stores it
        in `self.image`.
        """

        if self.image is not None:
            return

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

#########################
# IMPORT GT ANNOTATIONS #
#########################

    def load_annotation_from_csv(self,
                                 annotations_folder: str,
                                 is_ground_truth=True):
        """
        Load figure annotations from the given (individual) csv file.

        Args:
            annotations_folder: Path to an annotation file (csv format).
            is_ground_truth:    Tells whether annotations are ground truth or predictions
                                    If True, annotations will be stored in `self.gt_panels`
                                    else, in `self.pred_panels`
        """

        base_name = os.path.splitext(self.image_filename)[0]

        annotation_csv = os.path.join(annotations_folder, base_name)

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
                    raise ValueError("Row should be of length 6 or 11")

                image_path = row[0]
                panel_coordinates = [int(x) for x in row[1:5]]
                panel_class = row[5]
                assert panel_class == 'panel'

                assert image_path == self.image_path, "Wrong image path in csv:"\
                    "\n\timage file name : {}"\
                    "\n\timage in csv row : {}".format(self.image_path,
                                                       image_path)

                # Instanciate Panel object
                panel = Panel(panel_rect=panel_coordinates,
                              label_rect=label_coordinates,
                              label=label)

                panels.append(panel)

        if is_ground_truth:
            self.gt_panels = panels
        else:
            self.pred_panels = panels



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
                    label_text = label_class.map_label(words[1])

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
                A list of Panel objects representing the detected labels.
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
                label_text = label_class.map_label(label_text)

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
                self._logger.error('%s: has unknown <shape> xml items %s',
                                   annotation_file_path, text)

        # Extract information from and validate all panel items
        panels = extract_panel_info()

        # Extract information from and validate all label items
        labels = extract_label_info()

        # Match both lists to get a unique list of panels containing
        #   information of their matching label
        # Save this list of panels in tha appropriate attribute of
        #   the figure object
        self.gt_panels = match_panels_with_labels(panels=panels,
                                                  labels=labels)


##############
# EVALUATION #
##############

    def get_num_correct_predictions(self,
                                    use_overlap_instead_of_iou=False,
                                    threshold: float = None) -> int:
        """
        Compute the number of rightly predicted panels for the figure according to oen of the
        following criteria:
            * A predicted panel which has an IoU > `threshold` (0.5 by default) with a
                ground truth panel is counted as a positive match (default case, i.e. when
                `use_overlap_instead_of_iou` is False).
                => This method is the one to choose to later compute mAP.

            * A predicted panel which has an overlap > `threshold` (0.66 by default) with a
                ground truth panel is counted as a positive match (when
                `use_overlap_instead_of_iou` is True)
                => This method is the one to choose to later compute the ImageCLEF accuracy.
                (see http://ceur-ws.org/Vol-1179/CLEF2013wn-ImageCLEF-SecoDeHerreraEt2013b.pdf)

        Args:
            use_overlap_instead_of_iou (bool):  if True, computes the number of correct matches
                                                    using the ImageCLEF rule. This affects the
                                                    default value of the threshold.
            threshold (float):                  The iou (or overlap) threshold needed for a
                                                    prediction to be classified as valid.

        Returns:
            num_correct (int): The number of accurate predictions.
        """

        if threshold is None:
            threshold = 0.66 if use_overlap_instead_of_iou else 0.5

        num_correct = 0
        picked_pred_panels_indices = [False for _ in range(len(self.pred_panels))]
        for gt_panel in self.gt_panels:
            max_metric = -1
            best_matching_pred_panel_index = -1

            for pred_panel_index, pred_panel in enumerate(self.pred_panels):
                if picked_pred_panels_indices[pred_panel_index]:
                    continue

                if use_overlap_instead_of_iou:
                    # --> Using ImageCLEF metric
                    intersection_area = box.intersection_area(gt_panel.panel_rect,
                                                              pred_panel.panel_rect)
                    if intersection_area == 0:
                        continue
                    pred_panel_area = box.area(pred_panel.panel_rect)
                    overlap = intersection_area / pred_panel_area

                    metric = overlap
                else:
                    # Using IoU (common for object detection)
                    iou = box.iou(gt_panel.panel_rect, pred_panel.panel_rect)

                    metric = iou

                if metric > max_metric:
                    max_metric = metric
                    best_matching_pred_panel_index = pred_panel_index

            if max_metric > threshold:
                num_correct += 1
                picked_pred_panels_indices[best_matching_pred_panel_index] = True

        return num_correct


##################
# PREVIEW FIGURE #
##################

    def get_preview(self, mode='gt'):
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
            preview_img: the preview image
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

            if self.pred_panels is None:
                self._logger.warning("No predicted panels exist for this figure." \
                    " Hence, they cannot be displayed.")
                return preview_img

            for panel in self.pred_panels:
                # Yellow for predicted panels
                panel.draw_elements(image=preview_img,
                                    color=(255, 255, 0))

            return preview_img

        elif mode == 'gt':
            panels = self.gt_panels

        # mode = 'pred'
        else:
            panels = self.pred_panels

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


    def show_preview(self, mode='gt', delay=0, window_name=None):
        """
        Display a preview of the image along with the panels and labels drawn on top.

        Args:
            mode: Select which information to display:
                    * 'gt': only the ground truth
                    * 'pred': only the predictions
                    * 'both': both predicted and ground truth annotations.
            delay:       The number of seconds after which the window is closed
                if 0, the delay is disabled.
            window_name: Name of the image display window.
        """

        image_preview = self.get_preview(mode)

        if window_name is None:
            window_name = self.image_filename

        cv2.imshow(window_name, image_preview)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()


    def save_preview(self, folder: str, mode='gt'):
        """
        Save the annotation preview at folder.

        Args:
            folder (str): The folder where to store the image preview.
            mode (str): Select which information to display:
                        * 'gt': only the ground truth
                        * 'pred': only the predictions
                        * 'both': both predicted and ground truth annotations.
        """
        preview_img = self.get_preview(mode)


        # Remove extension from original figure image file name
        file_name = os.path.splitext(
            self.image_filename)[0]

        export_path = os.path.join(folder, file_name + "_preview.jpg")

        # Write the preview image file to destination
        cv2.imwrite(export_path, preview_img)


#################
# EXPORT FIGURE #
#################

    def export_gt_annotation_to_individual_csv(self,
                                               csv_export_dir: str = None):
        """
        Export the ground truth annotation of the figure to an individual csv file.

        Args:
            csv_export_dir: path to the directory where to export the csv file.
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
            for panel in self.gt_panels:

                csv_row = [
                    self.image_path,
                    panel.panel_rect[0],
                    panel.panel_rect[1],
                    panel.panel_rect[2],
                    panel.panel_rect[3],
                    'panel'
                    ]

                if panel.label is not None and panel.label_rect is not None:
                    csv_row.append(panel.label_rect[0])
                    csv_row.append(panel.label_rect[1])
                    csv_row.append(panel.label_rect[2])
                    csv_row.append(panel.label_rect[3])
                    csv_row.append(panel.label)

                # Writting to csv file
                csv_writer.writerow(csv_row)


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
