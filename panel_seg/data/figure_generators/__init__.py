"""
TODO
"""

from .global_csv_figure_generator import GlobalCsvFigureGenerator
from .image_clef_xml_figure_generator import ImageClefXmlFigureGenerator
from .iphotodraw_xml_figure_generator import IphotodrawXmlFigureGenerator
from .image_list_figure_generator import ImageListFigureGenerator

__all__ = [k for k in globals().keys() if not k.startswith("_")]
