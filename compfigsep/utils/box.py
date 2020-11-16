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


#############################################################################
Functions to deal with coordinates and areas of union, intersection of boxes.

In every 'box' function, boxes should respect the format [x1, y1, x2, y2].
"""
from typing import NamedTuple

from typing import Tuple
Box = Tuple[int, int, int, int]

class Point(NamedTuple):
    """
    NamedTuple representing a point in a 2D image.
    """
    x: int
    y: int


def get_center(box: Box) -> Point:
    """
    Compute the coordinates of the box center.

    Args:
        box (Box):  A box [x1, y1, x2, y2].

    Returns:
        x_center, y_center (Point): The coordinates of the center.
    """
    x_center: float = (box[0] + box[2]) / 2.0
    y_center: float = (box[1] + box[3]) / 2.0

    return Point(int(x_center), int(y_center))


def union(box_1: Box,
          box_2: Box) -> Box:
    """
    Compute the coordinates of the smallest box containing both box_1 and box_2.

    Args:
        box_1 (Box):    box [x1, y1, x2, y2].
        box_2 (Box):    box [x1, y1, x2, y2].

    Returns:
        union (Box):    The union of box_1 and box_2 [x, y, x+w, y+h].
    """

    x_union: int = min(box_1[0], box_2[0])
    y_union: int = min(box_1[1], box_2[1])
    width_union: int = max(box_1[2], box_2[2]) - x_union
    height_union: int = max(box_1[3], box_2[3]) - y_union

    return (x_union,
            y_union,
            x_union + width_union,
            y_union + height_union)


def intersection(box_1: Box,
                 box_2: Box) -> Box:
    """
    Compute the coordinates of the intersection of ractangles box_1 and box_2.

    Args:
        box_1 (Box):    box [x1, y1, x2, y2].
        box_2 (Box):    box [x1, y1, x2, y2].

    Returns:
        intersection (Box): The intersection of box_1 and box_2
                                [x, y, x+w, y+h].
    """

    # box_1 and box_2 should be [x1, y1, x2, y2]
    x_intersect = max(box_1[0], box_2[0])
    y_intersect = max(box_1[1], box_2[1])
    width_intersect = min(box_1[2], box_2[2]) - x_intersect
    height_intersect = min(box_1[3], box_2[3]) - y_intersect

    if width_intersect < 0 or height_intersect < 0:
        return (0, 0, 0, 0)

    return (x_intersect,
            y_intersect,
            x_intersect + width_intersect,
            y_intersect + height_intersect)


def area(box: Box) -> int:
    """
    Compute the area of the given box.

    Args:
        box (Box):  box [x1, y1, x2, y2].

    Returns:
        area (int): The area of the box.
    """

    return (box[2] - box[0]) * (box[3] - box[1])


def union_area(box_1: Box,
               box_2: Box,
               area_intersection: int) -> int:
    """
    Compute the area of the union of the ractangles box_1 and box_2.

    Args:
        box_1 (Box):    box [x1, y1, x2, y2].
        box_2 (Box):    box [x1, y1, x2, y2].
        area_intersection (float):  The area of the intersection of box_1 and box_2.

    Returns:
        union_area (int):   The area of the union of box_1 and box_2.
    """

    area_1: int = area(box_1)
    area_2: int = area(box_2)

    area_union: int = area_1 + area_2 - area_intersection

    return area_union


def intersection_area(box_1: Box,
                      box_2: Box) -> int:
    """
    Compute the area of the intersection between box_1 and box_2.

    Args:
        box_1 (Box):    box [x1,y1,x2,y2].
        box_2 (Box):    box [x1,y1,x2,y2].

    Returns:
        intersection_area (int):    The area of the intersection of box_1 and box_2.
    """

    x_intersect: int = max(box_1[0], box_2[0])
    y_intersect: int = max(box_1[1], box_2[1])
    width_intersect: int = min(box_1[2], box_2[2]) - x_intersect
    height_intersect: int = min(box_1[3], box_2[3]) - y_intersect

    if width_intersect < 0 or height_intersect < 0:
        return 0

    return width_intersect * height_intersect


def iou(box_1: Box,
        box_2: Box) -> float:
    """
    Compute the 'intersection over union' (IoU) between box_1 and box_2.

    Args:
        box_1 (Box):    box [x1,y1,x2,y2].
        box_2 (Box):    box [x1,y1,x2,y2].

    Returns:
        iou (float):    The IoU between box_1 and box_2.
    """

    if box_1[0] >= box_1[2] \
            or box_2[1] >= box_2[3] \
            or box_2[0] >= box_2[2] \
            or box_2[1] >= box_2[3]:
        return 0.0

    area_i: int = intersection_area(box_1, box_2)
    area_u: int = union_area(box_1, box_2, area_i)

    return float(area_i) / float(area_u + 1e-6)
