"""
TODO
"""


def view_data_set(
        figure_generator):
    """
    Preview all the figures from a data set.
    The image is displayed along with the bounding boxes (panels and, if present, labels).

    Args:
        figure_generator: a generator of Figure objects.
    """

    for figure in figure_generator:
        figure.show_preview()
