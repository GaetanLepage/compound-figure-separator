"""
A class and an exception handling panels (part of a compound figure).
"""

from typing import List

import cv2

class Panel:
    """
    A class for a Panel (a subpart of a compound figure)

    Attributes:
        label: The panel's label
        panel_rect: The rectangle localizing the panel
        label_rect: The rectangle localizing the label
    """

    def __init__(
            self,
            panel_rect: List[float],
            label: str = None,
            label_rect: List[float] = None):
        """
        Init for a `Panel` object

        Args:
            label: the label of the Panel
            panel_rect: The rectangle localizing the panel
            label_rect: The rectangle localizing the label
        """

        self.label = label
        # list [x_min, y_min, x_max, y_max]
        self.panel_rect = panel_rect
        self.label_rect = label_rect

    def draw_elements(self, image, color):
        """
        Draw the panel bounding box and (if applicable) its associated label bounding box.
        This function does not return anything but affect the given image by side-effect.

        Args:
            image: The base image that will be used as a background.
            color: The color of the drawn elements.
        """
        # Draw panel box
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
                        org=(self.label_rect[2] + 10, self.label_rect[3]),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=color)


class PanelSegError(Exception):
    """
    Exception for FigureSeg

    Attributes:
        message
    """
