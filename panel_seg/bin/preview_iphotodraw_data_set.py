"""
TODO
"""

import sys
import argparse

sys.path.append(".")

from panel_seg.io.figure_generators import iphotodraw_xml_figure_generator
from panel_seg.io.figure_viewer import view_data_set


def parse_args(args):
    """
    TODO
    """
    parser = argparse.ArgumentParser(
        description='Preview all the figures from an iPhotoDraw data set.')

    parser.add_argument(
        '--eval_list_txt',
        help='The path to the txt file listing the images.',
        default='data/zou/eval.txt',
        type=str)

    parser.add_argument(
        '--image_directory_path',
        help='The path to the directory whre the images are stored.',
        default=None,
        type=str)

    return parser.parse_args(args)


def main(args=None):
    """
    TODO
    """

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    figure_generator = iphotodraw_xml_figure_generator(
        eval_list_txt=args.eval_list_txt,
        image_directory_path=args.image_directory_path)

    view_data_set(
        figure_generator=figure_generator,
        delay=100,
        window_name="PanelSeg data preview")


if __name__ == '__main__':
    main()
