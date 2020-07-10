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


#################################################
Figure generator handling a JSON annotation file.
"""

import os
import logging
import json
import re
from time import strptime
import datetime
import progressbar

import compfigsep

from ...utils.figure import (
    Figure,
    SubFigure,
    DetectedSubFigure,
    Panel,
    Label)

from .figure_generator import FigureGenerator

PROJECT_DIR = os.path.join(
    os.path.dirname(compfigsep.__file__),
    os.pardir)


def get_most_recent_json(folder_path: str = None) -> str:
    """
    Finds the most recent JSON annotation file within the given folder.

    Args:
        folder_path (str):  The path to the directory where to look for a JSON file.

    Returns:
        json_path (str):    The path to the most recent JSON annotation file.
    """
    default_path = os.path.join(PROJECT_DIR, 'output')

    if folder_path is None:
        folder_path = default_path
        logging.info(f"No folder_path given: using default : {default_path}")

    elif not os.path.isdir(folder_path):
        logging.info(f"Given folder_path {folder_path} is not a valid directory."\
                      " Using default: {default_path}")
        folder_path = default_path

    if not os.path.isdir(folder_path):
        logging.error(f"folder_path {folder_path} does not exist. Aborting.")
        return

    regexp_time_stamp_file_names = \
        r".*_[0-9]{4}-[A-Za-z]+-[0-3][0-9]_[0-2][0-9]:[0-5][0-9]:[0-5][0-9].json"

    dates = {}

    for file_name in os.listdir(folder_path):

        if not file_name.endswith('.json'):
            continue

        if re.search(pattern=regexp_time_stamp_file_names,
                     string=file_name) is None:
            continue

        date_string = re.search(pattern=r"[0-9]{4}-[A-Za-z]+-[0-3][0-9]_"\
                                         "[0-2][0-9]:[0-5][0-9]:[0-5][0-9]",
                                string=file_name).group(0)

        year = int(re.search(pattern=r"[0-9]{4}",
                             string=date_string).group(0))

        month_string = re.search(pattern=r"[A-Za-z]+",
                                 string=date_string).group(0)

        month_number = strptime(month_string, '%B').tm_mon

        day_number = int(re.search(pattern=r"-[0-3][0-9]_",
                                   string=date_string).group(0)[1:-1])

        time_string = re.search(pattern=r"[0-2][0-9]:[0-5][0-9]:[0-5][0-9]",
                                string=date_string).group(0)

        hour, minute, second = (int(value) for value in time_string.split(':'))

        date = datetime.datetime(year=year,
                                 month=month_number,
                                 day=day_number,
                                 hour=hour,
                                 minute=minute,
                                 second=second)

        dates[date] = file_name

    assert len(dates) > 0, f"No valid json annotation file was found in folder {folder_path}."\
                            "\nExiting"

    max_date = max(dates)

    most_recent_file_name = dates[max_date]

    return os.path.join(folder_path, most_recent_file_name)








class JsonFigureGenerator(FigureGenerator):
    """
    Generator of Figure objects from a JSON annotation file.

    Attributes:
        json_annotation_file_path (str):    The path to the json annotation file.
    """

    def __init__(self, json_annotation_file_path: str):
        """
        Init function.
        Call the init function of the abstract parent class.

        Args:
            json_annotation_file_path (str):    The path to the json annotation file.
        """

        self.json_annotation_file_path = json_annotation_file_path

        super().__init__()

        if not os.path.isfile(json_annotation_file_path):
            raise FileNotFoundError("The annotation json file does not exist :"\
                "\n\t {}".format(json_annotation_file_path))


    def __call__(self) -> Figure:
        """
        Generator of Figure objects from a json annotation file.

        Yields:
            figure (Figure):    Figure objects with annotations.
        """


        with open(self.json_annotation_file_path, 'r') as json_annotation_file:
            data_dict = json.load(json_annotation_file)

        for index, figure_dict in enumerate(progressbar.progressbar(data_dict.values())):


            yield Figure.from_dict(figure_dict=figure_dict,
                                   index=index)
