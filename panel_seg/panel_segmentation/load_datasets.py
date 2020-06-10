"""
Load PanelSeg data set to be used with the Detectron API for the panel segmentation task.
"""

from detectron2.data import DatasetCatalog, MetadataCatalog

from panel_seg.data.figure_generators import IphotodrawXmlFigureGenerator

# TODO remove
from panel_seg.data.figure_generators import ImageListFigureGenerator

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

    # Create the figure generator to feed the dictionary
    figure_generator = IphotodrawXmlFigureGenerator(
        eval_list_txt=eval_list_txt)

    # TODO remove
    # if dataset_name == "pubmed":
        # eval_list_txt = "data/pubmed_extract/eval_list.txt"
        # image_directory_path = "data/pubmed_extract/image/data/dmli-from-comp-reclassification-captions-prostate/"

    # figure_generator = ImageListFigureGenerator(eval_list_txt, image_directory_path)

    DatasetCatalog.register(name=dataset_name,
                            func=lambda: export_figures_to_detectron_dict(
                                figure_generator=figure_generator(),
                                task='panel_seg'))

    MetadataCatalog.get(name=dataset_name).set(figure_generator=figure_generator())

    # TODO remove if it is indeed useless
    # Add the class names as metadata.
    # MetadataCatalog.get(name=dataset_name).set(thing_classes=list(LABEL_CLASS_MAPPING.keys()))
