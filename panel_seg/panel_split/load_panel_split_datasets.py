#!/usr/bin/env python3

"""
Load ImageCLEF data set to be used with the Detectron API

TODO : refactor and make more generic.
"""

from detectron2.data import DatasetCatalog, MetadataCatalog

from panel_seg.io.figure_generators import (
    image_clef_xml_figure_generator,
    iphotodraw_xml_figure_generator)

from panel_seg.io.export import export_figures_to_detectron_dict


def register_panel_splitting_dataset(name):
    """
    TODO

    TODO: manage validation
    TODO: get detectron logger and WARN if dataset name is not valid
    """

    # ImageCLEF dataset
    if "image_clef" in name:

        if name == "image_clef_train":

            xml_annotation_file_path = "data/ImageCLEF/training/FigureSeparationTraining2016GT.xml"
            image_directory_path = "data/ImageCLEF/training/FigureSeparationTraining2016/"

        elif name == "image_clef_test":
            xml_annotation_file_path = "data/ImageCLEF/test/FigureSeparationTest2016GT.xml"
            image_directory_path = "data/ImageCLEF/test/FigureSeparationTest2016/"

        else:
            pass

        DatasetCatalog.register(name=name,
                                func=export_figures_to_detectron_dict(
                                    figure_generator=image_clef_xml_figure_generator(
                                        xml_annotation_file_path=xml_annotation_file_path,
                                        image_directory_path=image_directory_path)))

        MetadataCatalog.get(name=name).set(xml_annotation_file_path=xml_annotation_file_path)
        MetadataCatalog.get(name=name).set(image_directory_path=image_directory_path)


    # Dataset from Zou
    elif "panel_seg" in name:
        if name == "panel_seg_train":
            eval_list_txt = "data/zou/train.txt"
            image_directory_path = "data/zou/"

        elif name == "panel_seg_test":
            eval_list_txt = "data/zou/train.txt"
            image_directory_path = "data/zou/"

        else:
            pass

        DatasetCatalog.register(name=name,
                                func=export_figures_to_detectron_dict(
                                    figure_generator=iphotodraw_xml_figure_generator(
                                        eval_list_txt=eval_list_txt,
                                        image_directory_path=image_directory_path)))

        MetadataCatalog.get(name=name).set(eval_list_txt=eval_list_txt)
        MetadataCatalog.get(name=name).set(image_directory_path=image_directory_path)

    else:
        pass

    MetadataCatalog.get(name=name).set(thing_classes=["panel"])
