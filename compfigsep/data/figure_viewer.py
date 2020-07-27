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


####################################################################################
Use a figure generator for previewing a data set by displaying each image one by one
along with the annotations.
"""

from argparse import ArgumentParser

from ..data.figure_generators import FigureGenerator


def add_viewer_args(parser: ArgumentParser) -> None:
    """
    Add to the given parser the arguments relative to the preview options :
        * mode ('gt', 'pred' or 'both')
        * delay
        * save_preview

    Args:
        parser (ArgumentParser):    An ArgumentParser.
    """

    parser.add_argument('--mode',
                        help="mode: Select which information to display:"\
                            " ['gt': only the ground truth,"\
                            " 'pred': only the predictions,"\
                            " 'both': both predicted and ground truth annotations]",
                        default='both')

    parser.add_argument('--delay',
                        help="The number of seconds after which the window is closed."\
                                " If 0, the delay is disabled.",
                        type=int,
                        default=100)

    parser.add_argument('--save_preview',
                        help="Save the image previews in image files.",
                        action='store_true')


def view_data_set(figure_generator: FigureGenerator,
                  mode: str = 'both',
                  *,
                  save_preview: bool = False,
                  preview_folder: str = None,
                  delay: int = 0,
                  window_name: str = None) -> None:
    """
    Preview all the figures from a data set.
    The image is displayed along with the bounding boxes (panels and, if present, labels).

    Args:
        figure_generator (FigureGenerator): A generator of Figure objects.
        mode (str):                         Select which information to display:
                                                * 'gt': only the ground truth
                                                * 'pred': only the predictions
                                                * 'both': both predicted and ground truth
                                                            annotations.
        save_preview (bool):                If true, saves the preview as an image.
        preview_folder (str):               The path to the folder where to store the preview
                                                images.
        delay (int):                        The number of seconds after which the window is
                                                closed if 0, the delay is disabled.
        window_name (str):                  Name of the image display window.
    """

    for figure in figure_generator():
        figure.show_preview(mode=mode,
                            delay=delay,
                            window_name=window_name)

        if save_preview:
            figure.save_preview(folder=preview_folder)
