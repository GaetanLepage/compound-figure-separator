"""
TODO
"""

import sys
import argparse

sys.path.append(".")

from panel_seg.io.imageclef_input import show_imageclef_data_set

def parse_args(args):
    """
    TODO
    """
    parser = argparse.ArgumentParser(
        description='Preview all the figures from an ImageCLEF data set.')

    parser.add_argument(
        '--annotation_xml',
        help='The path to the xml annotation file.',
        default='data/imageCLEF/test/FigureSeparationTest2016GT.xml',
        type=str)

    parser.add_argument(
        '--image_directory_path',
        help='The path to the directory whre the images are stored.',
        default='data/imageCLEF/test/FigureSeparationTest2016/',
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

    show_imageclef_data_set(
        xml_annotation_file_path=args.annotation_xml,
        image_directory_path=args.image_directory_path)


if __name__ == '__main__':
    main()
