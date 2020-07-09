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


###################################
Classes panels and detected panels.
"""

from typing import Tuple, Dict

import cv2
import numpy as np

from ..box import Box

Color = Tuple[int, int, int]

DEFAULT_GT_COLOR = (0, 255, 0)
DEFAULT_DETECTION_COLOR = (0, 0, 200)


class Panel:
    """
    TODO

    Attributes:
        box (Box):  The bounding box localizing the panel.
    """

    def __init__(self, box: Box = None):
        """
        Init for a Panel.

        Args:
            box (Box):  The bounding box localizing the panel.
        """
        if isinstance(box, np.ndarray):
            box = box.tolist()

        self.box = [round(val) for val in box]


    @classmethod
    def from_dict(cls, panel_dict: Dict) -> 'Panel':
        """
        Instanciate a Panel object from a dictionnary.

        Args:
            panel_dict (Dict):  A dictionnary representing the panel information.

        Returns:
            panel (Panel):  The resulting Panel object.
        """

        return Panel(box=panel_dict.get('box'))


    def to_dict(self) -> Dict:
        """
        Export to a dict.

        Returns:
            output_dict (Dict): A Dict representing the panel information.
        """

        output_dict = {}

        if self.box is not None:
            output_dict['box'] = self.box

        return output_dict


    def draw(self,
             image: np.ndarray,
             color: Color = DEFAULT_GT_COLOR):
        """
        Draw the panel bounding box.
        This function does not return anything but affect the given image by side-effect.

        Args:
            image (np.ndarray): The base image that will be used as a background.
            color (Color):      The color of the drawn elements (in RGB format).
        """
        # Set default color if needed.
        if color is None:
            color = DEFAULT_GT_COLOR

        # Draw the panel box if it exists.
        if self.box is not None:
            cv2.rectangle(img=image,
                          pt1=(self.box[0], self.box[1]),
                          pt2=(self.box[2], self.box[3]),
                          color=color,
                          thickness=2)


    def __str__(self) -> str:
        """
        str method for a Panel.

        Returns:
            string (str):   A string containing the Panel information.
        """
        string = f"{type(self).__name__}:"
        string += f" box: {self.box}"

        return string


    def __repr__(self):
        return str(self)



class DetectedPanel(Panel):
    """
    TODO

    Attributes:
        box (Box):                          The bounding box localizing the panel.
        detection_score (float):            Detection score (confidence).
        is_true_positive_iou (bool):        Whether this is a correct panel detection (panel
                                                splitting and panel segmentation tasks).
        is_true_positive_overlap (bool):    Whether this is a correct panel detection (ImageCLEF
                                                panel splitting criteria).
    """

    def __init__(self,
                 box: Box = None,
                 detection_score: float = None):
        """
        Init for a DetectedPanel.

        Args:
            box (Box):                  The bounding box localizing the panel.
            detection_score (float):    Detection score (confidence).
        """
        super().__init__(box=box)

        self.detection_score = detection_score

        self.is_true_positive_iou = None
        self.is_true_positive_overlap = None


    @classmethod
    def from_normal_panel(cls, panel: Panel) -> 'DetectedPanel':
        """
        TODO
        """
        # If it is already a DetectedPanel, no need to do anything.
        if isinstance(panel, DetectedPanel):
            return panel

        if not isinstance(panel, Panel):
            raise ValueError(f"Invalid type for panel: {type(panel)}")

        return DetectedPanel(box=panel.box)


    @classmethod
    def from_dict(cls, panel_dict: Dict) -> 'DetectedPanel':
        """
        Instanciate a DetectedPanel object from a dictionnary.

        Args:
            panel_dict (Dict):  A dictionnary representing the panel information.

        Returns:
            detected_panel (DetectedPanel): The resulting DetectedPanel object.
        """

        detected_panel = DetectedPanel(box=panel_dict.get('box'),
                                       detection_score=panel_dict.get('detection_score'))

        detected_panel.is_true_positive_overlap = panel_dict.get('is_true_positive_overlap')

        detected_panel.is_true_positive_iou = panel_dict.get('is_true_positive_iou')

        return detected_panel


    def to_dict(self) -> Dict:
        """
        Export to a dict.

        Returns:
            output_dict (Dict): A Dict representing the panel information.
        """

        # Call the method form Panel class.
        output_dict = super().to_dict()

        if self.box is not None:
            output_dict['box'] = self.box

        if self.detection_score is not None:
            output_dict['detection_score'] = self.detection_score

        if self.is_true_positive_overlap is not None:
            output_dict['is_true_positive_overlap'] = self.is_true_positive_overlap

        if self.is_true_positive_iou is not None:
            output_dict['is_true_positive_iou'] = self.is_true_positive_iou

        return output_dict


    def draw(self,
             image: np.ndarray,
             color: Color = DEFAULT_DETECTION_COLOR):
        """
        TODO
        """
        # Set default color if needed.
        if color is None:
            color = DEFAULT_DETECTION_COLOR

        super().draw(image, color)


    def __str__(self):
        """
        str method for a DetectedPanel.

        Returns:
            string (str):   A string containing the DetectedPanel information.
        """
        string = super().__str__()

        if self.detection_score is not None:
            string += f", detection_score: {self.detection_score}"

        if self.is_true_positive_overlap is not None:
            string += f", is_true_positive_overlap: {self.is_true_positive_overlap}"

        if self.is_true_positive_iou is not None:
            string += f", is_true_positive_iou: {self.is_true_positive_iou}"

        return string


    def __repr__(self):
        return str(self)
