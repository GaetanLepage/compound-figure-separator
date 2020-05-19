"""
Use a figrure generator for previewing a data set by displaying each image one by one
along with the annotations.
"""

from argparse import ArgumentParser

def parse_viewer_args(parser: ArgumentParser):
    """
    Parse the argument relative to the preview options :
        * mode ('gt', 'pred' or 'both')
        * delay

    Args:
        parser: an ArgumentParser.

    Returns:
        parser: the 'given' parser augmented with the options.
    """

    parser.add_argument('--mode',
                        help="mode: Select which information to display:"\
                            " ['gt': only the ground truth,"\
                            " 'pred': only the predictions,"\
                            " 'both': both predicted and ground truth annotations]",
                        default='gt')

    parser.add_argument('--delay',
                        help="The number of seconds after which the window is closed."\
                                " If 0, the delay is disabled.",
                        type=int,
                        default=100)

    return parser


def view_data_set(figure_generator,
                  mode='gt',
                  delay=0,
                  window_name: str = None):
    """
    Preview all the figures from a data set.
    The image is displayed along with the bounding boxes (panels and, if present, labels).

    Args:
        figure_generator: a generator of Figure objects.
        mode: Select which information to display:
                * 'gt': only the ground truth
                * 'pred': only the predictions
                * 'both': both predicted and ground truth annotations.
        delay:       The number of seconds after which the window is closed
            if 0, the delay is disabled.
        window_name: Name of the image display window.
    """

    for figure in figure_generator:
        figure.show_preview(mode=mode,
                            delay=delay,
                            window_name=window_name)
