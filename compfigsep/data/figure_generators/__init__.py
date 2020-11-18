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


###############################################################################
Implementation of multiple Figure generators handling the CompFigSep data sets.
"""
from .figure_generator import *
from .global_csv_figure_generator import *
from .image_clef_xml_figure_generator import *
from .image_list_figure_generator import *
from .iphotodraw_xml_figure_generator import *
from .json_figure_generator import *

__all__ = [k for k in globals().keys() if not k.startswith("_")]
