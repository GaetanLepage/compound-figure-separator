"""
Load ImageCLEF and PanelSeg data sets to be used with the Detectron API for the
panel splitting task.

TODO : refactor and make more generic.
"""

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
            # TODO warn the user in this case
            pass

        # TODO might be useful to implement generators as a class
        # Create two instances of the figure_generator so that one is given to the metadata
        figure_generator = image_clef_xml_figure_generator(
            xml_annotation_file_path=xml_annotation_file_path,
            image_directory_path=image_directory_path)

        figure_generator_copy = image_clef_xml_figure_generator(
            xml_annotation_file_path=xml_annotation_file_path,
            image_directory_path=image_directory_path)


    # Dataset from Zou
    elif "zou" in dataset_name:
        if dataset_name == "zou_panel_splitting_train":
            eval_list_txt = "data/zou/train.txt"
            image_directory_path = "data/zou/"

        elif dataset_name == "zou_panel_splitting_test":
            eval_list_txt = "data/zou/train.txt"
            image_directory_path = "data/zou/"

        else:
            # TODO warn the user in this case
            pass

        # Create two instances of the figure_generator so that one is given to the metadata
        figure_generator = iphotodraw_xml_figure_generator(
            eval_list_txt=eval_list_txt)
        figure_generator_copy = iphotodraw_xml_figure_generator(
            eval_list_txt=eval_list_txt)

    else:
        # TODO warn the user in this case
        pass

    DatasetCatalog.register(name=dataset_name,
                            func=lambda: export_figures_to_detectron_dict(
                                figure_generator=figure_generator))

    MetadataCatalog.get(name=dataset_name).set(figure_generator=figure_generator_copy)

    MetadataCatalog.get(name=dataset_name).set(thing_classes=["panel"])
