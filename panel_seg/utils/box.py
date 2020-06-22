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


##################################################################################
Functions to deal with coordinates and areas of union, intersection of rectangles.

In every 'box' function, rectangles should respect the format [x1, y1, x2, y2].
"""

from typing import Tuple
Rectangle = Tuple[int, int, int, int]
Point = Tuple[float, float]

def get_center(rectangle: Rectangle) -> Point:
    """
    Compute the coordinates of the rectangle center.

    Args:
        rectangle (Rectangle):  A rectangle [x1, y1, x2, y2].

    Returns:
        x_center, y_center (Point): The coordinates of the center.
    """
    return (rectangle[0] + rectangle[2]) / 2.0,\
           (rectangle[1] + rectangle[3]) / 2.0


def union(rectangle_1: Rectangle,
          rectangle_2: Rectangle) -> Rectangle:
    """
    Compute the coordinates of the smallest rectangle
    containing both rectangle_1 and rectangle_2.

    Args:
        rectangle_1 (Rectangle): rectangle [x1, y1, x2, y2].
        rectangle_2 (Rectangle): rectangle [x1, y1, x2, y2].

    Returns:
        union (Rectangle):  The union of rectangle_1 and rectangle_2 [x, y, x+w, y+h].
    """

    x_union = min(rectangle_1[0], rectangle_2[0])
    y_union = min(rectangle_1[1], rectangle_2[1])
    width_union = max(rectangle_1[2], rectangle_2[2]) - x_union
    height_union = max(rectangle_1[3], rectangle_2[3]) - y_union

    return [
        x_union,
        y_union,
        x_union + width_union,
        y_union + height_union]


def intersection(rectangle_1: Rectangle,
                 rectangle_2: Rectangle) -> Rectangle:
    """
    Compute the coordinates of the intersection of ractangles rectangle_1 and rectangle_2.

    Args:
        rectangle_1 (Rectangle):    rectangle [x1, y1, x2, y2].
        rectangle_2 (Rectangle):    rectangle [x1, y1, x2, y2].

    Returns:
        intersection (Rectangle):   The intersection of rectangle_1 and rectangle_2
                                        [x, y, x+w, y+h].
    """

    # rectangle_1 and rectangle_2 should be [x1, y1, x2, y2]
    x_intersect = max(rectangle_1[0], rectangle_2[0])
    y_intersect = max(rectangle_1[1], rectangle_2[1])
    width_intersect = min(rectangle_1[2], rectangle_2[2]) - x_intersect
    height_intersect = min(rectangle_1[3], rectangle_2[3]) - y_intersect

    if width_intersect < 0 or height_intersect < 0:
        return [0, 0, 0, 0]

    return [
        x_intersect,
        y_intersect,
        x_intersect + width_intersect,
        y_intersect + height_intersect]


def area(rectangle: Rectangle) -> float:
    """
    Compute the area of the given rectangle.

    Args:
        rectangle (Rectangle):  rectangle [x1, y1, x2, y2].

    Returns:
        area (float):   The area of the rectangle.
    """

    return (rectangle[2] - rectangle[0]) * (rectangle[3] - rectangle[1])


def union_area(rectangle_1: Rectangle,
               rectangle_2: Rectangle,
               area_intersection: float) -> float:
    """
    Compute the area of the union of the ractangles rectangle_1 and rectangle_2.

    Args:
        rectangle_1 (Rectangle):    rectangle [x1, y1, x2, y2].
        rectangle_2 (Rectangle):    rectangle [x1, y1, x2, y2].
        area_intersection (float):  The area of the intersection of rectangle_1 and rectangle_2.

    Returns:
        union_area (float): The area of the union of rectangle_1 and rectangle_2.
    """

    area_1 = area(rectangle_1)
    area_2 = area(rectangle_2)

    area_union = area_1 + area_2 - area_intersection

    return area_union


def intersection_area(rectangle_1: Rectangle,
                      rectangle_2: Rectangle) -> float:
    """
    Compute the area of the intersection between rectangle_1 and rectangle_2.

    Args:
        rectangle_1 (Rectangle):    rectangle [x1,y1,x2,y2].
        rectangle_2 (Rectangle):    rectangle [x1,y1,x2,y2].

    Returns:
        intersection_area (float):  The area of the intersection of rectangle_1 and rectangle_2.
    """

    x_intersect = max(rectangle_1[0], rectangle_2[0])
    y_intersect = max(rectangle_1[1], rectangle_2[1])
    width_intersect = min(rectangle_1[2], rectangle_2[2]) - x_intersect
    height_intersect = min(rectangle_1[3], rectangle_2[3]) - y_intersect

    if width_intersect < 0 or height_intersect < 0:
        return 0

    return width_intersect * height_intersect


def iou(rectangle_1: Rectangle,
        rectangle_2: Rectangle) -> float:
    """
    Compute the 'intersection over union' (IoU) between rectangle_1 and rectangle_2.

    Args:
        rectangle_1 (Rectangle):    rectangle [x1,y1,x2,y2].
        rectangle_2 (Rectangle):    rectangle [x1,y1,x2,y2].

    Returns:
        iou (float):    The IoU between rectangle_1 and rectangle_2.
    """

    if rectangle_1[0] >= rectangle_1[2] \
            or rectangle_2[1] >= rectangle_2[3] \
            or rectangle_2[0] >= rectangle_2[2] \
            or rectangle_2[1] >= rectangle_2[3]:
        return 0.0

    area_i = intersection_area(rectangle_1, rectangle_2)
    area_u = union_area(rectangle_1, rectangle_2, area_i)

    return float(area_i) / float(area_u + 1e-6)
