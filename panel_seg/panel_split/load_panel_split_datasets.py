#!/usr/bin/env python3

"""
Load ImageCLEF data set to be used with the Detectron API
"""

from detectron2.data import DatasetCatalog, MetadataCatalog

from panel_seg.io.figure_generators import image_clef_xml_figure_generator
from panel_seg.io.export import export_figures_to_detectron_dict


DATASET_TRAIN_NAME = "image_clef_train"
TRAIN_XML = "data/ImageCLEF/training/FigureSeparationTraining2016-GT.xml"
TRAIN_IMAGE_PATH = "data/ImageCLEF/training/FigureSeparationTraining2016/"
DATASET_TEST_NAME = "image_clef_test"
TEST_XML = "data/ImageCLEF/test/FigureSeparationTest2016GT.xml"
TEST_IMAGE_PATH = "data/ImageCLEF/test/FigureSeparationTest2016/"

def _get_dicts_train():
    """
    Get the ImageCLEF training data set as a Python dict() compatible with Detectron2.

    Returns:
        training data set (dict)
    """
    train_figure_generator = image_clef_xml_figure_generator(
        xml_annotation_file_path=TRAIN_XML,
        image_directory_path=TRAIN_IMAGE_PATH)

    return export_figures_to_detectron_dict(train_figure_generator)


def _get_dicts_test():
    """
    Get the ImageCLEF test data set as a Python dict() compatible with Detectron2.

    Returns:
        test data set (dict)
    """

    test_figure_generator = image_clef_xml_figure_generator(
        xml_annotation_file_path=TEST_XML,
        image_directory_path=TEST_IMAGE_PATH)

    return export_figures_to_detectron_dict(test_figure_generator)


def register_image_clef_datasets():
    """
    Register the ImageCLEF dataset in the Detectron2 process to be used for training and testing.
    """

    # Register the training dataset
    DatasetCatalog.register(name=DATASET_TRAIN_NAME,
                            func=_get_dicts_train)
    MetadataCatalog.get(name=DATASET_TRAIN_NAME).set(thing_classes=["panel"])
    MetadataCatalog.get(name=DATASET_TRAIN_NAME).set(xml_annotation_file_path=TRAIN_XML)
    MetadataCatalog.get(name=DATASET_TRAIN_NAME).set(image_directory_path=TRAIN_IMAGE_PATH)

    # Register the test dataset
    DatasetCatalog.register(name=DATASET_TEST_NAME,
                            func=_get_dicts_test)
    MetadataCatalog.get(name=DATASET_TEST_NAME).set(thing_classes=["panel"])
    MetadataCatalog.get(name=DATASET_TEST_NAME).set(xml_annotation_file_path=TEST_XML)
    MetadataCatalog.get(name=DATASET_TEST_NAME).set(image_directory_path=TEST_IMAGE_PATH)
