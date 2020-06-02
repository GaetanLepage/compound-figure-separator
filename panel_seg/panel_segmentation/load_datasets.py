"""
Load PanelSeg data set to be used with the Detectron API for the panel segmentation task.
"""

from detectron2.data import DatasetCatalog, MetadataCatalog

from panel_seg.data.figure_generators import iphotodraw_xml_figure_generator

from panel_seg.data.export import export_figures_to_detectron_dict
from panel_seg.utils.figure.label_class import LABEL_CLASS_MAPPING


def register_panel_segmentation_dataset(dataset_name):
    """
    Register the appropriate data set for panel splitting in the Detectron `DatasetCatalog`.

    TODO: manage validation
    TODO: get detectron logger and WARN if dataset name is not valid

    Args:
        dataset_name (str): The name of the data set to register. Has to belong to accepted ones.
    """
    if not 'zou' in dataset_name:
        # TODO warn the user in this case
        pass

    # Dataset from Zou
    if dataset_name == "zou_panel_seg_train":
        eval_list_txt = "data/zou/train.txt"
        image_directory_path = "data/zou/"

    elif dataset_name == "zou_panel_seg_test":
        eval_list_txt = "data/zou/eval.txt"
        image_directory_path = "data/zou/"

    else:
        pass

    # Create two instances of the figure_generator so that one is given to the metadata
    figure_generator = iphotodraw_xml_figure_generator(
        eval_list_txt=eval_list_txt)
    figure_generator_copy = iphotodraw_xml_figure_generator(
        eval_list_txt=eval_list_txt)


    DatasetCatalog.register(name=dataset_name,
                            func=lambda: export_figures_to_detectron_dict(
                                figure_generator=figure_generator,
                                task='panel_seg'))

    MetadataCatalog.get(name=dataset_name).set(figure_generator=figure_generator_copy)

    # Add the class names as metadata.
    MetadataCatalog.get(name=dataset_name).set(thing_classes=list(LABEL_CLASS_MAPPING.keys()))
