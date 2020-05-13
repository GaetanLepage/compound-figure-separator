#!/usr/bin/env python3

"""
Load ImageCLEF data set to be used with the Detectron API

TODO : refactor and make more generic.
"""

from itertools import tee

from detectron2.data import DatasetCatalog, MetadataCatalog

from panel_seg.io.figure_generators import (
    image_clef_xml_figure_generator,
    iphotodraw_xml_figure_generator)

from panel_seg.io.export import export_figures_to_detectron_dict


def register_panel_splitting_dataset(dataset_name):
    """
    Register the appropriate data set for panel splitting in the Detectron `DatasetCatalog`.

    TODO: manage validation
    TODO: get detectron logger and WARN if dataset name is not valid

    Args:
        dataset_name (str): The name of the data set to register. Has to belong to accepted ones.
    """

    # ImageCLEF dataset
    if "image_clef" in dataset_name:

        if dataset_name == "image_clef_train":

            xml_annotation_file_path = "data/ImageCLEF/training/FigureSeparationTraining2016-GT.xml"
            image_directory_path = "data/ImageCLEF/training/FigureSeparationTraining2016/"

        elif dataset_name == "image_clef_test":
            xml_annotation_file_path = "data/ImageCLEF/test/FigureSeparationTest2016GT.xml"
            image_directory_path = "data/ImageCLEF/test/FigureSeparationTest2016/"

        elif dataset_name == "image_clef_validation":
            raise NotImplementedError("Validation data set has not yet been created.")

        else:
            pass

        # TODO might be useful to implement generators as a class
        # Create two instances of the figure_generator so that one is given to the metadata
        figure_generator, figure_generator_copy = tee(image_clef_xml_figure_generator(
            xml_annotation_file_path=xml_annotation_file_path,
            image_directory_path=image_directory_path))

        # MetadataCatalog.get(name=dataset_name).set(
            # xml_annotation_file_path=xml_annotation_file_path)
        # MetadataCatalog.get(name=dataset_name).set(
            # image_directory_path=image_directory_path)


    # Dataset from Zou
    elif "panel_seg" in dataset_name:
        if dataset_name == "panel_seg_train":
            eval_list_txt = "data/zou/train.txt"
            image_directory_path = "data/zou/"

        elif dataset_name == "panel_seg_test":
            eval_list_txt = "data/zou/train.txt"
            image_directory_path = "data/zou/"

        else:
            pass

        figure_generator, figure_generator_copy = tee(iphotodraw_xml_figure_generator(
            eval_list_txt=eval_list_txt))

        # MetadataCatalog.get(name=dataset_name).set(eval_list_txt=eval_list_txt)
        # MetadataCatalog.get(name=dataset_name).set(image_directory_path=image_directory_path)

    else:
        pass

    DatasetCatalog.register(name=dataset_name,
                            func=lambda: export_figures_to_detectron_dict(
                                figure_generator=figure_generator))

    MetadataCatalog.get(name=dataset_name).set(figure_generator=figure_generator_copy)

    MetadataCatalog.get(name=dataset_name).set(thing_classes=["panel"])