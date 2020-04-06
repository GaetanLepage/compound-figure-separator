"""
A class and an exception handling panels (part of a compound figure).
"""

from typing import List

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
            label: str,
            panel_rect: List[float],
            label_rect: List[float]):
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


class PanelSegError(Exception):
    """
    Exception for FigureSeg

    Attributes:
        message
    """
