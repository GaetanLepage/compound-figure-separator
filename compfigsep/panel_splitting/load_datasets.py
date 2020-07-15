"""
#############################
#        CompFigSep         #
# Compound Figure Separator #
#############################

GitHub:         https://github.com/GaetanLepage/compound-figure-separator

Author:         Gaétan Lepage
Email:          gaetan.lepage@grenoble-inp.org
Date:           Spring 2020

Master's project @HES-SO (Sierre, SW)

Supervisors:    Henning Müller (henning.mueller@hevs.ch)
                Manfredo Atzori (manfredo.atzori@hevs.ch)

Collaborator:   Niccolò Marini (niccolo.marini@hevs.ch)


###############################################################################
Load ImageCLEF and PanelSeg data sets to be used with the Detectron API for the
panel splitting task.
"""

from detectron2.data import DatasetCatalog, MetadataCatalog

from ..data.figure_generators import (
    ImageClefXmlFigureGenerator,
    IphotodrawXmlFigureGenerator)
from ..data.export import export_figures_to_detectron_dict


def register_panel_splitting_dataset(dataset_name: str):
    """
    Register the appropriate data set for panel splitting in the Detectron `DatasetCatalog`.

    Args:
        dataset_name (str): The name of the data set to register.
                                Has to belong to accepted ones.
    """
    # ImageCLEF dataset
    if "image_clef" in dataset_name:

        if dataset_name == "image_clef_train":
            xml_annotation_file_path = \
                "data/ImageCLEF/training/FigureSeparationTraining2016-GT.xml"
            image_directory_path = "data/ImageCLEF/training/FigureSeparationTraining2016/"

        elif dataset_name == "image_clef_test":
            xml_annotation_file_path = "data/ImageCLEF/test/FigureSeparationTest2016GT.xml"
            image_directory_path = "data/ImageCLEF/test/FigureSeparationTest2016/"

        elif dataset_name == "image_clef_validation":
            raise NotImplementedError("Validation data set has not yet been created.")

        else:
            # TODO warn the user in this case
            pass

        # Instanciate the FigureGenerator
        figure_generator = ImageClefXmlFigureGenerator(
            xml_annotation_file_path=xml_annotation_file_path,
            image_directory_path=image_directory_path)

    # PanelSeg data set
    elif "zou" in dataset_name:
        if dataset_name == "zou_panel_splitting_train":
            file_list_txt = "data/zou/train.txt"

        elif dataset_name == "zou_panel_splitting_test":
            file_list_txt = "data/zou/eval.txt"

        else:
            # TODO warn the user in this case
            pass

        # Instanciate the FigureGenerator
        figure_generator = IphotodrawXmlFigureGenerator(file_list_txt=file_list_txt)

    else:
        # TODO warn the user in this case
        pass

    # Register the data set and set the ingest function to convert Figures to Detectron dict.
    DatasetCatalog.register(name=dataset_name,
                            func=lambda: export_figures_to_detectron_dict(
                                figure_generator=figure_generator,
                                task='panel_splitting'))

    # Provide the figure generator to the DataSet for later evaluation.
    MetadataCatalog.get(name=dataset_name).set(figure_generator=figure_generator)

    # Add the class names (here, only 'panel') as metadata.
    MetadataCatalog.get(name=dataset_name).set(thing_classes=["panel"])
