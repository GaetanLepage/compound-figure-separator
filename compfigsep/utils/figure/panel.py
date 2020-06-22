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


#####################################################################
A class and an exception handling panels (part of a compound figure).
"""

from typing import List, Tuple

import cv2
import numpy as np

class Panel:
    """
    A class for a Panel (a subpart of a compound figure).

    Attributes:
        label (str):                The panel's label.
        panel_rect (List[float]):   The rectangle localizing the panel.
        label_rect (List[float]):   The rectangle localizing the label.
    """

    def __init__(self,
                 panel_rect: List[float],
                 label: str = None,
                 label_rect: List[float] = None):
        """
        Init for a `Panel` object

        Args:
            panel_rect (List[float]):   The rectangle localizing the panel.
            label (str):                The panel's label.
            label_rect (List[float]):   The rectangle localizing the label.
        """

        self.label = label
        # list [x_min, y_min, x_max, y_max]
        self.panel_rect = panel_rect
        self.label_rect = label_rect


    def __str__(self) -> str:
        """
        str method for a Panel object.

        Returns:
            string (str):   A pretty representation of the Panel informations.
        """
        string = f"{type(self).__name__}"

        if self.label is not None:
            string += f" label = {self.label}"
        if self.panel_rect is not None:
            string += f" | panel_rect = {self.panel_rect}"
        if self.label_rect is not None:
            string += f" | label_rect = {self.label_rect}"

        return string


    def draw_elements(self,
                      image: np.ndarray,
                      color: Tuple[int, int, int]):
        """
        Draw the panel bounding box and (if applicable) its associated label bounding box.
        This function does not return anything but affect the given image by side-effect.

        Args:
            image (np.ndarray):             The base image that will be used as a background.
            color (Tuple[int, int, int]):   The color of the drawn elements (in RGB format).
        """
        # Draw panel box
        if self.panel_rect is not None:
            cv2.rectangle(img=image,
                          pt1=(self.panel_rect[0], self.panel_rect[1]),
                          pt2=(self.panel_rect[2], self.panel_rect[3]),
                          color=color,
                          thickness=2)

        if self.label_rect is not None:
            # Draw label box
            cv2.rectangle(img=image,
                          pt1=(self.label_rect[0], self.label_rect[1]),
                          pt2=(self.label_rect[2], self.label_rect[3]),
                          color=color,
                          thickness=2)

            # Draw label text
            cv2.putText(img=image,
                        text=self.label,
                        org=(int(self.label_rect[2]) + 10,
                             int(self.label_rect[3])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=color)


    def add_label_info(self, label: 'Panel'):
        """
        Augment the Panel object by adding information from a label (given as a Panel object).

        Args:
            label (Panel):  A Panel object containing label information.
        """
        self.label_rect = label.label_rect
        self.label = label.label


class DetectedPanel(Panel):
    """
    A Panel subclass handling detected panels.
    Add association attributes (to link detections to ground truth elements).

    Attributes:
        label (str):                            The panel's label.
        panel_rect (List[float]):               The rectangle localizing the panel.
        label_rect (List[float]):               The rectangle localizing the label.
        panel_is_true_positive_iou (bool):      Whether this is a correct panel detection
                                                    (panel splitting and panel segmentation
                                                    tasks).
        panel_is_true_positive_overlap (bool):  Whether this is a correct panel detection
                                                    (ImageCLEF panel splitting criteria).
        panel_detection_score (float):          Panel detection score.
        label_is_true_positive (bool):          Whether this is a correct label detection
                                                    (label recognition task).
        label_detection_score (float):          Label detection score.
    """

    def __init__(self,
                 panel_rect: List[float] = None,
                 panel_detection_score: float = None,
                 label: str = None,
                 label_rect: List[float] = None,
                 label_detection_score: float = None):
        """
        Init for a `DetectedPanel` object.

        Args:
            panel_rect (List[float]):       The rectangle localizing the panel.
            panel_detection_score (float):  Panel detection score.
            label (str):                    The label of the Panel.
            label_rect (List[float]):       The rectangle localizing the label.
            panel_detection_score (float):  Panel detection score.
        """
        # Call the Panel init.
        super().__init__(panel_rect=panel_rect,
                         label=label,
                         label_rect=label_rect)

        self.panel_detection_score = panel_detection_score
        self.label_detection_score = label_detection_score

        self.panel_is_true_positive_iou = False
        self.panel_is_true_positive_overlap = False

        self.label_is_true_positive = False


    def add_label_info(self, label: 'DetectedPanel'):
        """
        Augment the Panel object by adding information from a label (given as a Panel object).

        Args:
            label (Panel):  A Panel object containing label information.
        """
        super().add_label_info(label)

        self.label_detection_score = label.label_detection_score


    def __str__(self) -> str:
        """
        str method for a DetectedPanel object.

        Returns:
            string (str):   A pretty representation of the Panel informations.
        """
        string = super().__str__()

        if self.panel_detection_score is not None:
            string += f" | panel_detection_score = {self.panel_detection_score}"
        if self.panel_is_true_positive_overlap is not None:
            string += f" | panel_is_true_positive_overlap = {self.panel_is_true_positive_overlap}"
        if self.panel_is_true_positive_iou is not None:
            string += f" | panel_is_true_positive_iou = {self.panel_is_true_positive_iou}"
        if self.label_detection_score is not None:
            string += f" | label_detection_score = {self.label_detection_score}"
        if self.label_is_true_positive is not None:
            string += f" | label_is_true_positive = {self.label_is_true_positive}"

        return string
