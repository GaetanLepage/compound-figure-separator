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


#####################################
Export tools for compfigseg datasets.
"""

import csv
import json
import os
import logging
import datetime

from ..utils.figure import Figure
from .figure_generators import FigureGenerator
from ..utils.figure.label import map_label, LABEL_CLASS_MAPPING


import compfigsep
PROJECT_DIR = os.path.join(os.path.dirname(compfigsep.__file__),
                           os.pardir)

def export_figures_to_csv(figure_generator: FigureGenerator,
                          output_csv_file: str,
                          individual_export: bool = False,
                          individual_export_csv_directory: str = None):
    """
    Export a set of figures to a csv file.
    This may be used for keras-retinanet.

    Args:
        figure_generator (FigureGenerator):     A generator yielding figure objects
        output_csv_file (str):                  The path of the csv file containing the
                                                    annotations
        individual_csv (bool):                  If True, also export the annotation to a single
                                                    csv file
        individual_export_csv_directory (str):  The path of the directory whete to store the
                                                    individual csv annotation files."
    """

    with open(output_csv_file, 'w', newline='') as csvfile:

        csv_writer = csv.writer(csvfile, delimiter=',')

        # Looping over Figure objects thanks to generator
        for figure in figure_generator:

            # Looping over SubFigure objects
            for subfigure in figure.gt_subfigures:

                panel = subfigure.panel
                csv_row = [
                    figure.image_path,
                    panel.box[0],
                    panel.box[1],
                    panel.box[2],
                    panel.box[3],
                    'panel'
                    ]

                label = subfigure.label
                if label is not None and label.box is not None:
                    csv_row.append(label.box[0])
                    csv_row.append(label.box[1])
                    csv_row.append(label.box[2])
                    csv_row.append(label.box[3])
                    csv_row.append(label.text)
                else:
                    csv_row += [''] * 5
                csv_writer.writerow(csv_row)

                if individual_export:
                    figure.export_annotation_to_individual_csv(
                        csv_export_dir=individual_export_csv_directory)


def export_figures_to_json(figure_generator: FigureGenerator,
                           json_output_filename: str = None,
                           json_output_directory: str = None):
    """
    Export a data set that can contain ground truth and/or detected annotations for any task to a
    JSON file.

    Args:
        figure_generator (FigureGenerator): A generator yielding figure objects.
        json_output_filename (str):         Name of the JSON output file.
        json_output_directory (str):        Path to the directory where to save the JSON
                                                output file.
    """

    if json_output_directory is None:
        json_output_directory = os.path.join(PROJECT_DIR, "output/")

    if not os.path.isdir(json_output_directory):
        os.mkdir(json_output_directory)

    if json_output_filename is None:
        json_output_filename = "compfigsep_experiment_{date:%Y-%B-%d_%H:%M:%S}.json".format(
            date=datetime.datetime.now())

    json_output_path = os.path.join(json_output_directory,
                                    json_output_filename)

    if os.path.isfile(json_output_path):
        logging.warning(f"JSON output file already exist ({json_output_path})."\
                         "\nAborting export.")

    output_dict = {}

    # Loop over the figure from the generator and add their dict representation to the output
    # dictionnary.
    for figure in figure_generator:
        output_dict[figure.image_filename] = figure.to_dict()

    with open (json_output_path, 'w') as json_file:
        json.dump(obj=output_dict,
                  fp=json_file,
                  indent=4)


def export_figures_to_tf_record(figure_generator: FigureGenerator,
                                tf_record_filename: str):
    """
    Convert a set of figures to a a TensorFlow records file.

    Args:
        figure_generator (FigureGenerator): A generator yielding figure objects.
        tf_record_filename (str):           Path to the output tf record file.
    """
    import tensorflow as tf

    with tf.io.TFRecordWriter(tf_record_filename) as writer:

        for figure in figure_generator():

            tf_example = figure.convert_to_tf_example()

            writer.write(tf_example.SerializeToString())


def export_figures_to_detectron_dict(figure_generator: FigureGenerator,
                                     task: str = 'panel_splitting') -> dict:
    """
    Export a set of Figure objects to a dict which is compatible with Facebook Detectron 2.

    Args:
        figure_generator (FigureGenerator): A generator yielding figure objects.
        task (str):                         The task for which to export the figures.

    Returns:
        dataset_dicts (dict): A dict representing the data set.
    """
    from detectron2.structures import BoxMode

    if task not in ['panel_splitting', 'label_recog', 'panel_seg']:
        raise ValueError("`task` has to be one of ['panel_splitting', 'label_recog',"\
                        f" 'panel_seg'] but is {task}")

    dataset_dicts = []
    for index, figure in enumerate(figure_generator()):
        record = {}

        record['file_name'] = figure.image_path
        record['image_id'] = index
        record['height'] = figure.image_height
        record['width'] = figure.image_width

        objs = []

        if figure.gt_subfigures is not None:

            for subfigure in figure.gt_subfigures:

                panel = subfigure.panel
                label = subfigure.label

                if task == 'panel_splitting':
                    obj = {
                        'bbox': panel.box,
                        'bbox_mode': BoxMode.XYXY_ABS,
                        'category_id': 0
                    }

                elif task == 'label_recog':

                    if label is None:
                        continue

                    if label.box is None or len(label.text) != 1:
                        # We ensure that, for this task, the labels are valid
                        # (they have been previously checked while loading annotations)
                        continue

                    obj = {
                        'bbox': label.box,
                        'bbox_mode': BoxMode.XYXY_ABS,
                        'category_id': LABEL_CLASS_MAPPING[label.text]
                    }

                # panel segmentation task
                else:
                    # category_id is artificially set to zero to satisfy the Detectron API.
                    # The actual label (if any) is stored in 'label'.
                    obj = {
                        'panel_bbox': panel.box,
                        'bbox_mode': BoxMode.XYXY_ABS
                    }

                    if label is not None\
                        and label.box is not None\
                        and len(label.text) == 1:
                        # If there is no valid label, it won't be considered for training.
                        # TODO: later, we would like to handle >1 length labels
                        obj['label_bbox'] = label.box
                        obj['label'] = LABEL_CLASS_MAPPING[map_label(label.text)]


                objs.append(obj)

            record["annotations"] = objs

        dataset_dicts.append(record)

    return dataset_dicts
