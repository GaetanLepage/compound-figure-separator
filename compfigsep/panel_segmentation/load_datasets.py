"""
#############################
#        CompFigSep         #
# Compound Figure Separator #
#############################

GitHub:         https://github.com/GaetanLepage/compound-figure-separator

Author:         Gaétan Lepage
Email:          gaetan.lepage@grenoble-inp.fr
Date:           Spring 2020

Master's project @HES-SO (Sierre, SW)

Supervisors:    Henning Müller (henning.mueller@hevs.ch)
                Manfredo Atzori (manfredo.atzori@hevs.ch)

Collaborators:  Niccolò Marini (niccolo.marini@hevs.ch)
                Stefano Marchesin (stefano.marchesin@unipd.it)


#########################################################################################
Load PanelSeg data set to be used with the Detectron API for the panel segmentation task.
"""

from detectron2.data import DatasetCatalog, MetadataCatalog

from ..data.figure_generators import IphotodrawXmlFigureGenerator

from ..data.export import export_figures_to_detectron_dict


def register_panel_segmentation_dataset(dataset_name):
    """
    Register the appropriate data set for panel splitting in the Detectron `DatasetCatalog`.

    Args:
        dataset_name (str): The name of the data set to register. Has to belong to accepted ones.
    """
    if not 'zou' in dataset_name:
        # TODO warn the user in this case
        return

    # Dataset from Zou
    if dataset_name == "zou_panel_seg_train":
        file_list_txt = "data/zou/train.txt"
        image_directory_path = "data/zou/"

    elif dataset_name == "zou_panel_seg_test":
        file_list_txt = "data/zou/eval.txt"
        image_directory_path = "data/zou/"

    else:
        return

    # Create the figure generator to feed the dictionary
    figure_generator = IphotodrawXmlFigureGenerator(
        file_list_txt=file_list_txt)


    # Register the data set and set the ingest function to convert Figures to Detectron dict.
    DatasetCatalog.register(name=dataset_name,
                            func=lambda: export_figures_to_detectron_dict(
                                figure_generator=figure_generator,
                                task='panel_seg'))

    MetadataCatalog.get(name=dataset_name).set(figure_generator=figure_generator)

    # TODO remove if it is indeed useless
    # Add the class names as metadata.
    # MetadataCatalog.get(name=dataset_name).set(thing_classes=list(LABEL_CLASS_MAPPING.keys()))
