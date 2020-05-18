"""
Export tools for panel-seg datasets
"""

import csv
import tensorflow as tf

from panel_seg.utils.figure.label_class import LABEL_CLASS_MAPPING

def export_figures_to_csv(figure_generator,
                          output_csv_file: str,
                          individual_export=False,
                          individual_export_csv_directory=None):
    """
    Export a set of figures to a csv file.
    This may be used for keras-retinanet.

    Args:
        figure_generator:                A generator yielding figure objects
        output_csv_file:                 The path of the csv file containing the annotations
        individual_csv:                  If True, also export the annotation to a single csv file
        individual_export_csv_directory: The path of the directory whete to store the individual
                                            csv annotation files."
    """

    with open(output_csv_file, 'w', newline='') as csvfile:

        csv_writer = csv.writer(csvfile, delimiter=',')

        # Looping over Figure objects thanks to generator
        for figure in figure_generator:

            # Looping over Panel objects
            for panel in figure.gt_panels:

                csv_row = [
                    figure.image_path,
                    panel.panel_rect[0],
                    panel.panel_rect[1],
                    panel.panel_rect[2],
                    panel.panel_rect[3],
                    'panel'
                    ]

                if panel.label is not None and panel.label_rect is not None:
                    csv_row.append(panel.label_rect[0])
                    csv_row.append(panel.label_rect[1])
                    csv_row.append(panel.label_rect[2])
                    csv_row.append(panel.label_rect[3])
                    csv_row.append(panel.label)
                else:
                    csv_row += ['']*5
                csv_writer.writerow(csv_row)

                if individual_export:
                    figure.export_annotation_to_individual_csv(
                        csv_export_dir=individual_export_csv_directory)


def export_figures_to_tf_record(figure_generator,
                                tf_record_filename):
    """
    Convert a set of figures to a a TensorFlow records file.

    Args:
        figure_generator:   A generator yielding figure objects.
        tf_record_filename: Path to the output tf record file.
    """

    with tf.io.TFRecordWriter(tf_record_filename) as writer:

        for figure in figure_generator:

            tf_example = figure.convert_to_tf_example()

            writer.write(tf_example.SerializeToString())


def export_figures_to_detectron_dict(figure_generator, task='panel_splitting'):
    """
    Export a set of Figure objects to a dict which is compatible with Facebook Detectron 2.

    Args:
        figure_generator:   A generator yielding figure objects.

    Returns:
        dataset_dicts (dict): a dict representing the data set.
    """
    from detectron2.structures import BoxMode

    if task not in ['panel_splitting', 'label_recog', 'panel_seg']:
        raise ValueError("`task` has to be one of ['panel_splitting', 'label_recog',"\
            f" 'panel_seg'] but is {task}")

    dataset_dicts = []
    for index, figure in enumerate(figure_generator):
        record = {}

        record["file_name"] = figure.image_path
        record["image_id"] = index
        record["height"] = figure.image_height
        record["width"] = figure.image_width

        objs = []

        for panel in figure.gt_panels:

            if task == 'panel_splitting':
                obj = {
                    "bbox": panel.panel_rect,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0
                }
            elif task == 'label_recog':
                if panel.label_rect is None or len(panel.label) != 1:
                    # We ensure that, for this task, the labels are valid
                    # (they have been previously checked while loading annotations)
                    continue

                obj = {
                    "bbox": panel.label_rect,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": LABEL_CLASS_MAPPING[panel.label]
                }
            # panel segmentation task
            else:
                raise NotImplementedError("Dict() export is not yet implemented.")

            objs.append(obj)

        record["annotations"] = objs

        if len(objs) != 0:
            dataset_dicts.append(record)

    return dataset_dicts
