"""
Deal with annotations from ImageCLEF xml files.
"""

import os
import argparse
import csv

from typing import Iterator


def parse_args(args):
    """
    TODO
    """
    parser = argparse.ArgumentParser(
        description='Convert ImageCLEF PanelSeg data set to CSV format.')

    parser.add_argument(
        '--data_root',
        help='The path to the directory where ImageCLEF images are stored.',
        default='data/ImageCLEF/training/',
        type=str)

    parser.add_argument(
        '--gt_path',
        help='The original ImageCLEF ground truth.XML file.',
        default='/datasets/ImageCLEF/2016/training/FigureSeparationTraining2016-GT.xml',
        type=str)

    parser.add_argument(
        '--csv_path',
        help='The converted csv file path.',
        default='train.csv',
        type=str)

    parser.add_argument('--list_path',
                        help='The figure image file list.',
                        default='train.txt',
                        type=str)

    return parser.parse_args(args)




def convert_annotations_to_csv(
        figure_iterator: Iterator,
        output_csv_file: str,
        output_list_file: str):
    """
    Convert annotations from Figure objects (yielded by `figure_iterator`) to a csv file.
    Also output a .txt file containing the list of image file names.

    Args:
        figure_iterator: TODO
        output_csv_file: the path where to store the newly created csv file
        output_list_file: TODO
    """
    image_path_list = list()

    with open(output_csv_file, 'w', newline='') as csvfile:

        # CSV writer
        csv_writer = csv.writer(csvfile, delimiter=',')

        for figure in figure_iterator:

            # Add the path of the image to the list of image files
            image_path_list.append(figure.image_path + "\n")

            image = read_image_bgr(image_path)
            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

            csv_path_individual = os.path.join('csv/', image_filename + '.csv')
            jpg_path_individual = os.path.join('preview/', image_filename + '.jpg')
            with open(csv_path_individual, 'w', newline='') as csvfile_individual:
                csv_writer_individual = csv.writer(csvfile_individual, delimiter=',')

                object_items = annotation_item.findall('./object')
                for idx, object_item in enumerate(object_items):
                    point_items = object_item.findall('./point')
                    x_1 = point_items[0].get('x')
                    y_1 = point_items[0].get('y')
                    x_2 = point_items[3].get('x')
                    y_2 = point_items[3].get('y')

                    if int(x_1) >= int(x_2) or int(y_1) >= int(y_2):
                        continue

                    csv_writer.writerow(
                        [image_path, x_1, y_1, x_2, y_2, 'panel'])

                    csv_writer_individual.writerow(
                        [image_path, x_1, y_1, x_2, y_2, 'panel'])

                    color = label_color(idx)
                    box = [
                        int(x_1),
                        int(y_1),
                        int(x_2),
                        int(y_2)
                        ]

                    draw_box(draw, box, color)

                cv2.imwrite(jpg_path_individual, draw)


        with open(args.list_path, "w") as text_file:
            text_file.writelines(image_paths)


def main():
    """
    TODO
    """

if __name__ == "__main__":
    main()
