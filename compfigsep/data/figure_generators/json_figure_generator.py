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


#################################################
Figure generator handling a JSON annotation file.
"""

from __future__ import annotations
import os
import logging
import json
import re
from time import strptime
import datetime
from typing import Iterable
from argparse import ArgumentParser

import progressbar # type: ignore

import compfigsep

from ...utils.figure import Figure

from .figure_generator import FigureGenerator

MODULE_DIR = os.path.dirname(compfigsep.__file__)
PROJECT_DIR = os.path.join(MODULE_DIR,
                           os.pardir)

DEFAULT_JSON_FOLDER = os.path.relpath(os.path.join(MODULE_DIR,
                                                   "compound_figure_separation/output/"))


def get_most_recent_json(folder_path: str = None) -> str:
    """
    Finds the most recent JSON annotation file within the given folder.

    Args:
        folder_path (str):  The path to the directory where to look for a JSON file.

    Returns:
        json_path (str):    The path to the most recent JSON annotation file.
    """
    default_path = DEFAULT_JSON_FOLDER

    if folder_path is None:
        folder_path = default_path
        logging.info("No folder_path given: using default : %s",
                     default_path)

    elif not os.path.isdir(folder_path):
        logging.warning("Given folder_path %s is not a valid directory."\
                        " Using default: %s",
                        folder_path, default_path)
        folder_path = default_path

    if not os.path.isdir(folder_path):
        logging.error("folder_path %s does not exist. Aborting.",
                      folder_path)

        raise ValueError(f"folder_path {folder_path} does not exist. Aborting.")

    regexp_time_stamp_file_names = \
        r".*_[0-9]{4}-[A-Za-z]+-[0-3][0-9]_[0-2][0-9]:[0-5][0-9]:[0-5][0-9].json"

    dates = {}

    for file_name in os.listdir(folder_path):

        if not file_name.endswith('.json'):
            continue

        if re.search(pattern=regexp_time_stamp_file_names,
                     string=file_name) is None:
            continue

        date_match = re.search(pattern=r"[0-9]{4}-[A-Za-z]+-[0-3][0-9]_"\
                                        "[0-2][0-9]:[0-5][0-9]:[0-5][0-9]",
                               string=file_name)
        if date_match is None:
            continue

        date_string = date_match.group(0)

        # Year
        year_match = re.search(pattern=r"[0-9]{4}",
                               string=date_string)
        if year_match is None:
            continue

        year = int(year_match.group(0))

        # Month
        month_match = re.search(pattern=r"[A-Za-z]+",
                                string=date_string)

        if month_match is None:
            continue

        month_string = month_match.group(0)

        month_number = strptime(month_string, '%B').tm_mon

        # Day
        day_match = re.search(pattern=r"-[0-3][0-9]_",
                              string=date_string)
        if day_match is None:
            continue

        day_number = int(day_match.group(0)[1:-1])

        # Time
        time_match = re.search(pattern=r"[0-2][0-9]:[0-5][0-9]:[0-5][0-9]",
                               string=date_string)
        if time_match is None:
            continue

        time_string = time_match.group(0)

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


def add_json_arg(parser: ArgumentParser,
                 json_default_relative_path: str = None,
                 folder_default_relative_path: str = None):
    """
    Parse the argument for loading a json file.

    Args:
        parser (ArgumentParser):            An ArgumentParser.
        json_default_relative_path (str):   Default relative (to MODULE_DIR) path to a json file.
        folder_default_relative_path (str): Default folder relative (to MODULE_DIR) path where to
                                                look for the most recent json file.
    """

    if json_default_relative_path is None:

        if folder_default_relative_path is None:
            folder_default_relative_path = "compound_figure_separation/output/"

        folder_default_path = os.path.join(MODULE_DIR,
                                           folder_default_relative_path)

        folder_default_path = os.path.relpath(folder_default_path)

        json_default_path = get_most_recent_json(folder_path=folder_default_path)

    else:
        json_default_path = os.path.join(MODULE_DIR,
                                         json_default_relative_path)

    json_default_path = os.path.relpath(json_default_path)

    parser.add_argument('--json',
                        help="The path to the json annotation file.",
                        default=json_default_path,
                        type=str)


class JsonFigureGenerator(FigureGenerator):
    """
    Generator of Figure objects from a JSON annotation file.

    Attributes:
        json_annotation_file_path (str):    The path to the json annotation file.
    """

    def __init__(self, json_path: str) -> None:
        """
        Args:
            json (str): The path to a json annotation file OR to a folder where to look for the
                            most recent file.
        """
        if os.path.isfile(json_path) and json_path.endswith('.json'):
            self.json_annotation_file_path = json_path

        elif os.path.isdir(json_path):
            self.json_annotation_file_path = get_most_recent_json(folder_path=json_path)

        print(self.json_annotation_file_path)

        super().__init__()

        if not os.path.isfile(self.json_annotation_file_path):
            raise FileNotFoundError("The annotation json file does not exist :"\
                "\n\t {}".format(self.json_annotation_file_path))


    def __copy__(self) -> JsonFigureGenerator:

        return JsonFigureGenerator(json_path=self.json_annotation_file_path)


    def __call__(self) -> Iterable[Figure]:
        """
        Generator of Figure objects from a json annotation file.

        Returns:
            Iterable[Figure]:   Figure objects with annotations.
        """

        with open(self.json_annotation_file_path, 'r') as json_annotation_file:
            data_dict = json.load(json_annotation_file)

        for index, figure_dict in enumerate(progressbar.progressbar(data_dict.values())):

            yield Figure.from_dict(figure_dict=figure_dict,
                                   index=index)
