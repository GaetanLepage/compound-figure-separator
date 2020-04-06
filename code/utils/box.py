from typing import List

"""
Functions to deal with coordinates and areas of union, intersection of rectangles.
"""

def union(
        rectangle_1: List[float],
        rectangle_2: List[float]):
    """
    Compute the coordinates of the smallest rectangle
    containing both rectangle_1 and rectangle_2.

    Args:
        rectangle_1: rectangle [x1,y1,x2,y2]
        rectangle_2: rectangle [x1,y1,x2,y2]

    Returns:
        The union of rectangle_1 and rectangle_2 [x, y, x+w, y+h]
    """

    x = min(rectangle_1[0], rectangle_2[0])
    y = min(rectangle_1[1], rectangle_2[1])
    w = max(rectangle_1[2], rectangle_2[2]) - x
    h = max(rectangle_1[3], rectangle_2[3]) - y

    return [x, y, x + w, y + h]


def intersection(
        rectangle_1: List[float],
        rectangle_2: List[float]):
    """
    Compute the coordinates of the ractangles rectangle_1 and rectangle_2.

    Args:
        rectangle_1: rectangle [x1,y1,x2,y2]
        rectangle_2: rectangle [x1,y1,x2,y2]

    Returns:
        The intersection of rectangle_1 and rectangle_2 [x, y, x+w, y+h]
    """

    # rectangle_1 and rectangle_2 should be [x1,y1,x2,y2]
    x = max(rectangle_1[0], rectangle_2[0])
    y = max(rectangle_1[1], rectangle_2[1])
    w = min(rectangle_1[2], rectangle_2[2]) - x
    h = min(rectangle_1[3], rectangle_2[3]) - y
    if w < 0 or h < 0:
        return [0, 0, 0, 0]
    return [x, y, x+w, y+h]


def union_area(
        rectangle_1: List[float],
        rectangle_2: List[float],
        area_intersection: float):
    """
    Compute the area of the union of the ractangles rectangle_1 and rectangle_2.

    Args:
        rectangle_1: rectangle [x1,y1,x2,y2]
        rectangle_2: rectangle [x1,y1,x2,y2]
        area_intersection: the area of the intersection of rectangle_1 and rectangle_2

    Returns:
        The area of the intersection of rectangle_1 and rectangle_2
    """

    area_1 = (rectangle_1[2] - rectangle_1[0]) * (rectangle_1[3] - rectangle_1[1])
    area_2 = (rectangle_2[2] - rectangle_2[0]) * (rectangle_2[3] - rectangle_2[1])
    area_union = area_1 + area_2 - area_intersection
    return area_union


def intersection_area(
        rectangle_1: List[float],
        rectangle_2: List[float]):
    """
    Compute the area of the ractangles rectangle_1 and rectangle_2.

    Args:
        rectangle_1: rectangle [x1,y1,x2,y2]
        rectangle_2: rectangle [x1,y1,x2,y2]

    Returns:
        The area of the intersection of rectangle_1 and rectangle_2
    """

    x = max(rectangle_1[0], rectangle_2[0])
    y = max(rectangle_1[1], rectangle_2[1])
    w = min(rectangle_1[2], rectangle_2[2]) - x
    h = min(rectangle_1[3], rectangle_2[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def iou(
        rectangle_1: List[float],
        rectangle_2: List[float]):
    """
    Compute the area of the ractangles rectangle_1 and rectangle_2.

    Args:
        rectangle_1: rectangle [x1,y1,x2,y2]
        rectangle_2: rectangle [x1,y1,x2,y2]

    Returns:
        The area of the intersection of rectangle_1 and rectangle_2
    """

    if rectangle_1[0] >= rectangle_1[2] \
            or rectangle_2[1] >= rectangle_2[3] \
            or rectangle_2[0] >= rectangle_2[2] \
            or rectangle_2[1] >= rectangle_2[3]:
        return 0.0

    area_i = intersection_area(rectangle_1, rectangle_2)
    area_u = union_area(rectangle_1, rectangle_2, area_i)

    return float(area_i) / float(area_u + 1e-6)


