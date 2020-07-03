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


##########################################
Classes for subfigures, panels and labels.
Also handles the detection information.
"""

from typing import Tuple

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
        self.box = box


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
        """
        TODO
        """
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

        self.is_true_positive_iou = False
        self.is_true_positive_overlap = False


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


    def __repr__(self):
        """
        TODO
        """
        return str(self)



class Label:
    """
    TODO

    Attributes:
        text (str): TODO
        box (Box):  TODO
    """

    def __init__(self,
                 text: str = None,
                 box: Box = None):
        """
        TODO
        """
        self.text = text
        self.box = box


    def draw(self,
             image: np.ndarray,
             color: Color = DEFAULT_GT_COLOR):
        """
        TODO
        """
        if color is None:
            color = DEFAULT_GT_COLOR
        # Draw label box
        if self.box is not None:
            cv2.rectangle(img=image,
                          pt1=(self.box[0], self.box[1]),
                          pt2=(self.box[2], self.box[3]),
                          color=color,
                          thickness=2)

        # Draw label text
        if self.text is not None:
            cv2.putText(img=image,
                        text=self.text,
                        org=(int(self.box[2]) + 10,
                             int(self.box[3])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=color)


    def __str__(self) -> str:
        """
        str method for a Label.

        Returns:
            string (str):   A string containing the Label information.
        """
        string = f"{type(self).__name__}:"
        string += f" box: {self.box}"
        string += f", text: {self.text}"

        return string


    def __repr__(self):
        """
        TODO
        """
        return str(self)



class DetectedLabel(Label):
    """
    TODO

    Attributes:
        text (str):                 TODO
        box (Box):                  TODO
        detection_score (float):    TODO
        is_true_positive (bool):    TODO
    """

    def __init__(self,
                 text: str = None,
                 box: Box = None,
                 detection_score: float = None):
        """
        TODO
        """
        super().__init__(text=text, box=box)

        self.detection_score = detection_score
        self.is_true_positive = False


    def draw(self,
             image: np.ndarray,
             color: Color = DEFAULT_DETECTION_COLOR):
        """
        TODO

        Args:
            image (np.ndarray): TODO
            color (Color):      TODO
        """
        if color is None:
            color = DEFAULT_DETECTION_COLOR

        super().draw(image, color)


    def __str__(self):
        """
        str method for a DetectedLabel.

        Returns:
            string (str):   A string containing the DetectedLabel information.
        """
        string = super().__str__()

        if self.detection_score is not None:
            string += f", detection_score: {self.detection_score}"

        if self.is_true_positive is not None:
            string += f", is_true_positive: {self.is_true_positive}"


    def __repr__(self) -> str:
        return str(self)



class SubFigure:
    """
    A class for a sub-figure (a subpart of a compound figure).

    Attributes:
        panel (Panel):  Panel object.
        label (Label):  Label object.
        caption (str):  Caption text.
    """

    def __init__(self,
                 panel: Panel = None,
                 label: Label = None,
                 caption: str = None):
        """
        Init for a `SubFigure` object.

        Args:
            panel (Panel):  Panel object.
            label (Label):  Label object.
            caption (str):  Caption text.
        """

        self.panel = panel
        self.label = label
        self.caption = caption


    def draw_elements(self,
                      image: np.ndarray,
                      color: Color = DEFAULT_GT_COLOR):
        """
        Draw the panel bounding box and (if applicable) its associated label bounding box.
        This function does not return anything but affect the given image by side-effect.

        Args:
            image (np.ndarray): The base image that will be used as a background.
            color (Color):      The color of the drawn elements (in RGB format).
        """
        if self.panel is not None:
            self.panel.draw(image=image, color=color)

        if self.label is not None:
            self.label.draw(image=image, color=color)


    def __str__(self) -> str:
        """
        str method for a SubFigure object.

        Returns:
            string (str):   A pretty representation of the SubFigure information.
        """
        string = f"{type(self).__name__}:"

        string += f" {self.panel}"
        string += f", {self.label}"
        string += f", Caption: {self.caption}"

        return string


    def __repr__(self) -> str:
        return str(self)



class DetectedSubFigure(SubFigure):
    """
    A SubFigure subclass handling detected sub-figures.
    Add association attributes (to link detections to ground truth elements).

    Attributes:
        panel (DetectedPanel):      Panel object.
        label (DetectedLabel):      Label object.
        caption (str):              Caption text.
        is_true_positive (bool):    TODO
    """

    def __init__(self,
                 panel: DetectedPanel = None,
                 label: DetectedLabel = None,
                 caption: str = None):
        """
        Init for a `DetectedPanel` object.

        Args:
        """
        # Call the SubFigure init.
        super().__init__(panel=panel,
                         label=label,
                         caption=caption)

        self.is_true_positive = False

        self.caption_is_positive = False


    def __str__(self) -> str:
        """
        str method for a DetectedSubFigure object.

        Returns:
            string (str):   A pretty representation of the SubFigure information.
        """
        string = super().__str__()

        string += f", is_true_positive: {self.is_true_positive}"
        string += f", caption_is_positive: {self.caption_is_positive}"

        return string


    def __repr__(self) -> str:
        return str(self)
