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
Classes Label and DetectedLabel.
"""

from __future__ import annotations
from typing import cast, Tuple, Dict, Optional

import cv2 # type: ignore
import numpy as np # type: ignore

from ...box import Box

Color = Tuple[int, int, int]

DEFAULT_GT_COLOR = (0, 255, 0)
DEFAULT_DETECTION_COLOR = (0, 0, 200)


class Label:
    """
    Class representing a label.

    Attributes:
        text (str): The label text ('A' or '1' or 'ii'...).
        box (Box):  The bounding box localizing the label on the image.
    """

    def __init__(self,
                 text: str = None,
                 box: Box = None) -> None:
        """
        Args:
            text (str): The label text ('A' or '1' or 'ii'...).
            box (Box):  The bounding box localizing the label on the image.
        """
        self.text = text

        if isinstance(box, np.ndarray):
            box = box.tolist()

        self.box = box

        if box is not None:
            self.box = cast(Box,
                            tuple([round(val) for val in box]))


    @classmethod
    def from_dict(cls, label_dict: Dict) -> Label:
        """
        Instanciate a Label object from a dictionnary.

        Args:
            label_dict (Dict):  A dictionnary representing the label information.

        Returns:
            label (Label):  The resulting Label object.
        """

        return Label(text=label_dict.get('text'),
                     box=label_dict.get('box'))


    def to_dict(self) -> Dict:
        """
        Export to a dict.

        Returns:
            output_dict (Dict): A Dict representing the label information.
        """

        output_dict: Dict = {}

        if self.text is not None:
            output_dict['text'] = self.text

        if self.box is not None:
            output_dict['box'] = self.box

        return output_dict


    def draw(self,
             image: np.ndarray,
             color: Color = DEFAULT_GT_COLOR) -> None:
        """
        Draw the label bounding box and text on the image.
        The image is affected by side-effect.

        Args:
            image (np.ndarray): Image to override with annotations.
            color (Color):      Color to draw the element with (in RGB format).
        """
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
        string = f"{type(self).__name__}:"
        string += f" box: {self.box}"
        string += f", text: {self.text}"

        return string


    def __repr__(self) -> str:
        return str(self)



class DetectedLabel(Label):
    """
    Class representing a detected label.

    Attributes:
        text (str):                 The label text ('A' or '1' or 'ii'...).
        box (Box):                  The bounding box localizing the label on the image.
        detection_score (float):    Detection score (confidence).
        is_true_positive (bool):    Whether this is a correct label detection.
    """

    def __init__(self,
                 text: str = None,
                 box: Box = None,
                 detection_score: float = None) -> None:
        """
        Args:
            text (str):                 The label text ('A' or '1' or 'ii'...).
            box (Box):                  The bounding box localizing the label on the image.
            detection_score (float):    Detection score (confidence).
        """
        super().__init__(text=text,
                         box=box)

        self.detection_score: Optional[float] = detection_score
        self.is_true_positive: bool = False


    @classmethod
    def from_normal_label(cls, label: Label) -> DetectedLabel:
        """
        Build a DetectedLabel object from a normal Label object.

        Args:
            label (Label):  A Label object.

        Returns:
            DetectedLabel:  The resulting DetectedLabel object.
        """
        # If it is already a DetectedLabel, no need to do anything.
        if isinstance(label, DetectedLabel):
            return label

        if not isinstance(label, Label):
            raise ValueError(f"Invalid type for label: {type(label)}")

        return DetectedLabel(text=label.text,
                             box=label.box)


    @classmethod
    def from_dict(cls, label_dict: Dict) -> DetectedLabel:
        """
        Instanciate a DetectedLabel object from a dictionnary.

        Args:
            label_dict (Dict):  A dictionnary representing the label information.

        Returns:
            detected_label (DetectedLabel): The resulting DetectedLabel object.
        """

        detected_label = DetectedLabel(text=label_dict.get('text'),
                                       box=label_dict.get('box'),
                                       detection_score=label_dict.get('detection_score'))

        if 'is_true_positive' in label_dict:
            detected_label.is_true_positive = label_dict['is_true_positive']

        return detected_label


    def to_dict(self) -> Dict:
        """
        Export to a dict.

        Returns:
            output_dict (Dict): A Dict representing the panel information.
        """

        # Call the method form Panel class.
        output_dict: Dict = super().to_dict()

        if self.box is not None:
            output_dict['box'] = self.box

        if self.detection_score is not None:
            output_dict['detection_score'] = self.detection_score

        if self.is_true_positive is not None:
            output_dict['is_true_positive'] = self.is_true_positive

        return output_dict


    def draw(self,
             image: np.ndarray,
             color: Color = DEFAULT_DETECTION_COLOR) -> None:
        """
        Draw the label bounding box and text on the image.
        the image is affected by side-effect.

        args:
            image (np.ndarray): image to override with annotations.
            color (color):      color to draw the element with.
        """
        super().draw(image, color)


    def __str__(self) -> str:
        string = super().__str__()

        if self.detection_score is not None:
            string += f", detection_score: {self.detection_score}"

        if self.is_true_positive is not None:
            string += f", is_true_positive: {self.is_true_positive}"

        return string


    def __repr__(self) -> str:
        return str(self)
