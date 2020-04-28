"""
TODO
"""


def view_data_set(figure_generator,
                  delay=0,
                  window_name=None):
    """
    Preview all the figures from a data set.
    The image is displayed along with the bounding boxes (panels and, if present, labels).

    Args:
        figure_generator: a generator of Figure objects.
    """

    for figure in figure_generator:
        figure.show_preview(delay=delay,
                            window_name=window_name)
