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
Implementation of multiple Figure generators handling the CompFigSep data sets.
"""
from .figure_generator import DATA_DIR
from .global_csv_figure_generator import GlobalCsvFigureGenerator
from .image_clef_xml_figure_generator import ImageClefXmlFigureGenerator
from .iphotodraw_xml_figure_generator import IphotodrawXmlFigureGenerator
from .image_list_figure_generator import ImageListFigureGenerator

__all__ = [k for k in globals().keys() if not k.startswith("_")]
