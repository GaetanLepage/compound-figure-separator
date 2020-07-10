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

from typing import Tuple, Dict

import cv2
import numpy as np

from ..box import Box

Color = Tuple[int, int, int]

DEFAULT_GT_COLOR = (0, 255, 0)
DEFAULT_DETECTION_COLOR = (0, 0, 200)


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

        if isinstance(box, np.ndarray):
            box = box.tolist()
        self.box = [round(val) for val in box]


    @classmethod
    def from_dict(cls, label_dict: Dict) -> 'Label':
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

        output_dict = {}

        if self.text is not None:
            output_dict['text'] = self.text

        if self.box is not None:
            output_dict['box'] = self.box

        return output_dict


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
        self.is_true_positive = None


    @classmethod
    def from_normal_label(cls, label: Label) -> 'DetectedLabel':
        """
        TODO
        """
        # If it is already a DetectedLabel, no need to do anything.
        if isinstance(label, DetectedLabel):
            return label

        if not isinstance(label, Label):
            raise ValueError(f"Invalid type for label: {type(label)}")

        return DetectedLabel(text=label.text,
                             box=label.box)

    @classmethod
    def from_dict(cls, label_dict: Dict) -> 'DetectedLabel':
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

        detected_label.is_true_positive = label_dict.get('is_true_positive')

        return detected_label


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

        if self.is_true_positive is not None:
            output_dict['is_true_positive'] = self.is_true_positive

        return output_dict


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

        return string


    def __repr__(self) -> str:
        return str(self)