"""
Use a figrure generator for previewing a data set by displaying each image one by one
along with the annotations.
"""


def view_data_set(figure_generator,
                  delay=0,
                  window_name: str = None):
    """
    Preview all the figures from a data set.
    The image is displayed along with the bounding boxes (panels and, if present, labels).

    Args:
        figure_generator: a generator of Figure objects.
        delay (int):
        window_name (str)
    """

    for figure in figure_generator:
        figure.show_preview(delay=delay,
                            window_name=window_name)
