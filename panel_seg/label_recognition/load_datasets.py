"""
Load PanelSeg data set to be used with the Detectron API for the label recognition task.
"""

from detectron2.data import DatasetCatalog, MetadataCatalog

from panel_seg.data.figure_generators import IphotodrawXmlFigureGenerator

from panel_seg.data.export import export_figures_to_detectron_dict
from panel_seg.utils.figure.label_class import LABEL_CLASS_MAPPING


def register_label_recognition_dataset(dataset_name: str):
    """
    Register the appropriate data set for label recognition in the Detectron `DatasetCatalog`.

    TODO: manage validation
    TODO: get detectron logger and WARN if dataset name is not valid

    Args:
        dataset_name (str): The name of the data set to register.
                                Has to belong to accepted ones.
    """
    # TODO: manage validation
    # TODO: get detectron logger and WARN if dataset name is not valid

    if not 'zou' in dataset_name:
        # TODO warn the user in this case
        pass

    # Dataset from Zou
    if dataset_name == "zou_label_recog_train":
        eval_list_txt = "data/zou/train.txt"

    elif dataset_name == "zou_label_recog_test":
        eval_list_txt = "data/zou/eval.txt"

    else:
        # TODO warn the user in this case
        pass

    # Create a first instance of the figure generator for the data set samples to be loaded.
    figure_generator = IphotodrawXmlFigureGenerator(
        eval_list_txt=eval_list_txt)

    # Register the data set.
    DatasetCatalog.register(name=dataset_name,
                            func=lambda: export_figures_to_detectron_dict(
                                figure_generator=figure_generator(),
                                task='label_recog'))

    MetadataCatalog.get(name=dataset_name).set(figure_generator=figure_generator())

    # Add the class names as metadata.
    MetadataCatalog.get(name=dataset_name).set(thing_classes=list(LABEL_CLASS_MAPPING.keys()))
