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

Collaborators:  Niccolò Marini (niccolo.marini@hevs.ch)
                Stefano Marchesin (stefano.marchesin@unipd.it)


################################
Classes Panel and DetectedPanel.
"""

from __future__ import annotations
from typing import cast, Tuple, Dict, Optional, Any

import cv2 # type: ignore
import numpy as np # type: ignore

from ..box import Box

Color = Tuple[int, int, int]

DEFAULT_GT_COLOR = (0, 255, 0)
DEFAULT_DETECTION_COLOR = (0, 0, 200)


class Panel:
    """
    Class representing a lanel.

    Attributes:
        box (Box):  The bounding box localizing the panel on the image.
    """

    def __init__(self, box: Box):
        """
        Args:
            box (Box):  The bounding box localizing the panel on the image.
        """
        if isinstance(box, np.ndarray):
            box = box.tolist()

        self.box = cast(Box,
                        tuple([round(val) for val in box]))


    @classmethod
    def from_dict(cls, panel_dict: Dict) -> Panel:
        """
        Instanciate a Panel object from a dictionnary.

        Args:
            panel_dict (Dict):  A dictionnary representing the panel information.

        Returns:
            panel (Panel):  The resulting Panel object.
        """
        return Panel(box=panel_dict['box'])


    def to_dict(self) -> Dict:
        """
        Export to a dict.

        Returns:
            output_dict (Dict): A Dict representing the panel information.
        """
        output_dict: Dict[str, Any] = {}

        if self.box is not None:
            output_dict['box'] = self.box

        return output_dict


    def draw(self,
             image: np.ndarray,
             color: Color = DEFAULT_GT_COLOR) -> None:
        """
        Draw the panel bounding box.
        The image is affected by side-effect.

        Args:
            image (np.ndarray): The base image that will be used as a background.
            color (Color):      Color to draw the element with (in RGB format).
        """
        # Draw the panel box if it exists.
        if self.box is not None:
            cv2.rectangle(img=image,
                          pt1=(self.box[0], self.box[1]),
                          pt2=(self.box[2], self.box[3]),
                          color=color,
                          thickness=2)


    def __str__(self) -> str:
        string = f"{type(self).__name__}:"
        string += f" box: {self.box}"

        return string


    def __repr__(self) -> str:
        return str(self)



class DetectedPanel(Panel):
    """
    Class representing a detected panel.

    Attributes:
        box (Box):                          The bounding box localizing the panel on the image.
        detection_score (float):            Detection score (confidence).
        is_true_positive_iou (bool):        Whether this is a correct panel detection (panel
                                                splitting and panel segmentation tasks).
        is_true_positive_overlap (bool):    Whether this is a correct panel detection (ImageCLEF
                                                panel splitting criteria).
    """

    def __init__(self,
                 box: Box,
                 detection_score: float = None) -> None:
        """
        Args:
            box (Box):                  The bounding box localizing the panel on the image.
            detection_score (float):    Detection score (confidence).
        """
        super().__init__(box=box)

        self.detection_score = detection_score

        self.is_true_positive_iou: Optional[bool] = None
        self.is_true_positive_overlap: Optional[bool] = None


    @classmethod
    def from_normal_panel(cls, panel: Panel) -> DetectedPanel:
        """
        Build a DetectedPanel object from a normal Panel object.

        Args:
            panel (Panel):  A Panel object.

        Returns:
            DetectedPanel:  The resulting DetectedPanel object.
        """
        # If it is already a DetectedPanel, no need to do anything.
        if isinstance(panel, DetectedPanel):
            return panel

        if not isinstance(panel, Panel):
            raise ValueError(f"Invalid type for panel: {type(panel)}")

        return DetectedPanel(box=panel.box)


    @classmethod
    def from_dict(cls, panel_dict: Dict) -> DetectedPanel:
        """
        Instanciate a DetectedPanel object from a dictionnary.

        Args:
            panel_dict (Dict):  A dictionnary representing the panel information.

        Returns:
            detected_panel (DetectedPanel): The resulting DetectedPanel object.
        """

        if 'box' in panel_dict:
            detected_panel = DetectedPanel(box=panel_dict['box'],
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
             color: Color = DEFAULT_DETECTION_COLOR
             ) -> None:
        """
        Draw the panel bounding box on the image.
        the image is affected by side-effect.

        args:
            image (np.ndarray): image to override with annotations.
            color (color):      color to draw the element with.
        """
        # Set default color if needed.
        if color is None:
            color = DEFAULT_DETECTION_COLOR

        super().draw(image, color)


    def __str__(self) -> str:
        string = super().__str__()

        if self.detection_score is not None:
            string += f", detection_score: {self.detection_score}"

        if self.is_true_positive_overlap is not None:
            string += f", is_true_positive_overlap: {self.is_true_positive_overlap}"

        if self.is_true_positive_iou is not None:
            string += f", is_true_positive_iou: {self.is_true_positive_iou}"

        return string


    def __repr__(self) -> str:
        return str(self)
