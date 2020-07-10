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
from . import Panel, DetectedPanel, Label, DetectedLabel

Color = Tuple[int, int, int]

DEFAULT_GT_COLOR = (0, 255, 0)
DEFAULT_DETECTION_COLOR = (0, 0, 200)


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


    @classmethod
    def from_dict(cls, sub_figure_dict: Dict) -> 'SubFigure':
        """
        Instanciate a SubFigure object from a dictionnary.

        Args:
            sub_figure_dict (Dict): A dictionnary representing the sub-figure information.

        Returns:
            sub_figure (SubFigure): The resulting SubFigure object.
        """
        panel = None
        if 'panel' in sub_figure_dict:
            panel = Panel.from_dict(sub_figure_dict['panel'])

        label = None
        if 'label' in sub_figure_dict:
            label = Label.from_dict(sub_figure_dict['label'])


        return SubFigure(panel=panel,
                         label=label,
                         caption=sub_figure_dict.get('caption'))


    def to_dict(self) -> Dict:
        """
        Export to a dict.

        Returns:
            output_dict (Dict): A Dict representing the sub-figure information.
        """

        output_dict = {}

        if self.panel is not None:
            output_dict['panel'] = self.panel.to_dict()

        if self.label is not None:
            output_dict['label'] = self.label.to_dict()

        if self.caption is not None:
            output_dict['caption'] = self.caption

        return output_dict


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


    @classmethod
    def from_normal_sub_figure(cls, subfigure: SubFigure) -> 'DetectedSubFigure':
        """
        TODO
        """
        # If it is already a DetectedSubFigure, no need to do anything.
        if isinstance(subfigure, DetectedSubFigure):
            return subfigure

        if subfigure.panel is None:
            panel = None
        else:
            panel = DetectedPanel.from_normal_panel(subfigure.panel)

        if subfigure.label is None:
            label = None
        else:
            label = DetectedLabel.from_normal_label(subfigure.label)

        return DetectedSubFigure(panel=panel,
                                 label=label,
                                 caption=subfigure.caption)


    @classmethod
    def from_dict(cls, detected_sub_figure_dict: Dict) -> 'DetectedSubFigure':
        """
        Instanciate a DetectedSubFigure object from a dictionnary.

        Args:
            detected_sub_figure_dict (Dict):    A dictionnary representing the sub-figure
                                                    information.

        Returns:
            detected_sub_figure (DetectedSubFigure):    The resulting DetectedSubFigure object.
        """
        panel = None
        if 'panel' in detected_sub_figure_dict:
            panel = DetectedPanel.from_dict(detected_sub_figure_dict['panel'])

        label = None
        if 'label' in detected_sub_figure_dict:
            panel = DetectedLabel.from_dict(detected_sub_figure_dict['label'])


        detected_sub_figure = DetectedSubFigure(panel=panel,
                                                label=label,
                                                caption=detected_sub_figure_dict.get('caption'))

        detected_sub_figure.is_true_positive = detected_sub_figure_dict.get('is_true_positive')

        detected_sub_figure.caption_is_positive = \
            detected_sub_figure_dict.get('caption_is_positive')

        return detected_sub_figure


    def to_dict(self) -> Dict:
        """
        Export to a dict.

        Returns:
            output_dict (Dict): A Dict representing the sub-figure information.
        """

        output_dict = super().to_dict()

        if self.is_true_positive is not None:
            output_dict['is_true_positive'] = self.is_true_positive

        if self.caption_is_positive is not None:
            output_dict['caption_is_positive'] = self.caption_is_positive

        return output_dict


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
