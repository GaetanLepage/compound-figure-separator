"""
TODO
"""

import sys
import argparse

from code.io.iphotodraw_input import show_iphotodraw_data_set

def parse_args(args):
    """
    TODO
    """
    parser = argparse.ArgumentParser(
        description='Preview all the figures from an iPhotoDraw data set.')

    parser.add_argument(
        '--eval_list_csv',
        help='The path to the csv file listing the images.',
        default='data/zou/eval.csv',
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

    show_iphotodraw_data_set(
        eval_list_csv=args.eval_list_csv,
        image_directory_path=args.image_directory_path)


if __name__ == '__main__':
    main()
