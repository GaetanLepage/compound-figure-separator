"""
A class and an exception handling panels (part of a compound figure).
"""

from typing import List

import cv2

class Panel:
    """
    A class for a Panel (a subpart of a compound figure).

    Attributes:
        label (str):                The panel's label
        panel_rect (List[float]):   The rectangle localizing the panel
        label_rect (List[float]):   The rectangle localizing the label
    """

    def __init__(self,
                 panel_rect: List[float],
                 label: str = None,
                 label_rect: List[float] = None):
        """
        Init for a `Panel` object

        Args:
            panel_rect (List[float]):   The rectangle localizing the panel
            label (str):                The panel's label
            label_rect (List[float]):   The rectangle localizing the label
        """

        self.label = label
        # list [x_min, y_min, x_max, y_max]
        self.panel_rect = panel_rect
        self.label_rect = label_rect

    def __str__(self):
        """
        TODO
        """
        return f"{type(self).__name__} label = {self.label}"\
                f" | panel_rect = {self.panel_rect} | label_rect = {self.label_rect}"

    def draw_elements(self, image, color):
        """
        Draw the panel bounding box and (if applicable) its associated label bounding box.
        This function does not return anything but affect the given image by side-effect.

        Args:
            image: The base image that will be used as a background.
            color: The color of the drawn elements.
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


class DetectedPanel(Panel):
    """
    TODO
    """

    def __init__(self,
                 panel_rect: List[float] = None,
                 panel_detection_score: float = None,
                 label: str = None,
                 label_rect: List[float] = None,
                 label_detection_score: float = None):
        """
        Init for a `DetectedPanel` object

        Args:
            label: the label of the Panel
            panel_rect: The rectangle localizing the panel
            label_rect: The rectangle localizing the label
        """

        super().__init__(panel_rect,
                         label,
                         label_rect)

        self.panel_detection_score = panel_detection_score
        self.label_detection_score = label_detection_score

        self.panel_is_true_positive_iou = False
        self.panel_is_true_positive_overlap = False

        self.label_is_true_positive = False


class PanelSegError(Exception):
    """
    Exception for FigureSeg

    Attributes:
        message
    """
