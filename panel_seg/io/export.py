"""
Export tools for panel-seg datasets
"""

import csv
import tensorflow as tf

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
            for panel in figure.panels:

                csv_row = [
                    figure.image_path,
                    panel.panel_rect[0],
                    panel.panel_rect[1],
                    panel.panel_rect[2],
                    panel.panel_rect[3],
                    'panel'
                    ]

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
