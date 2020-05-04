#!/usr/bin/env python3

"""
TODO
"""

from panel_seg.io.figure_generators import image_clef_xml_figure_generator

from panel_seg.io.export import export_figures_to_detectron_dict

from detectron2.data import DatasetCatalog, MetadataCatalog



DATASET_TRAIN_NAME = "image_clef_train"
DATASET_TEST_NAME = "image_clef_test"

def get_dicts_train():

    train_figure_generator = image_clef_xml_figure_generator(
        xml_annotation_file_path="data/ImageCLEF/training/FigureSeparationTraining2016-GT.xml",
        image_directory_path="data/ImageCLEF/training/FigureSeparationTraining2016/")

    return export_figures_to_detectron_dict(train_figure_generator)

def get_dicts_test():

    test_figure_generator = image_clef_xml_figure_generator(
        xml_annotation_file_path="data/ImageCLEF/test/FigureSeparationTest2016GT.xml",
        image_directory_path="data/ImageCLEF/test/FigureSeparationTest2016/")

    return export_figures_to_detectron_dict(test_figure_generator)

def register_image_clef_datasets():

    print("Loading image clef datastet")

    DatasetCatalog.register(DATASET_TRAIN_NAME,
                            get_dicts_train)
    MetadataCatalog.get(DATASET_TRAIN_NAME).set(thing_classes=["panel"])

    DatasetCatalog.register(DATASET_TEST_NAME,
                            get_dicts_test)
    MetadataCatalog.get(DATASET_TEST_NAME).set(thing_classes=["panel"])

