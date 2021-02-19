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


###############################
Defines a structure for labels.
"""

from __future__ import annotations

import operator
from enum import Enum
from typing import List, Dict, Callable

import numpy as np

from .utils import (is_lower_char,
                    is_upper_char,
                    UC_ROMAN,
                    LC_ROMAN,
                    UC_ROMAN_TO_INT,
                    LC_ROMAN_TO_INT)


class LabelStructureEnum(Enum):
    """
    Enum representing the category of labels for a figure.
    """

    # The image contain no labels
    NONE = 0

    # 1, 2, 3...
    NUMERICAL = 1

    # A, B, C
    LATIN_UC = 2

    # a, b, c
    LATIN_LC = 3

    # I, II, III
    ROMAN_UC = 4

    # i, ii, iii
    ROMAN_LC = 5

    OTHER = 6


LABEL_FILTER: Dict[LabelStructureEnum, Callable[[str], bool]] = {
    LabelStructureEnum.NONE: lambda char: char == '_',
    LabelStructureEnum.NUMERICAL: lambda char: char.isdigit(),
    LabelStructureEnum.LATIN_UC: is_upper_char,
    LabelStructureEnum.LATIN_LC: is_lower_char,
    LabelStructureEnum.ROMAN_UC: lambda char: char in UC_ROMAN,
    LabelStructureEnum.ROMAN_LC: lambda char: char in LC_ROMAN
}

# Used to map a label to its 'index' (i.e. its numerical value):
# 'A' --> 1
# '1' --> 1
# 'v' --> 5
LABEL_INDEX: Dict[LabelStructureEnum, Callable[[str], int]] = {
    LabelStructureEnum.NONE: lambda char: -1,
    LabelStructureEnum.NUMERICAL: int,
    LabelStructureEnum.LATIN_UC: lambda char: ord(char) - 64,
    LabelStructureEnum.LATIN_LC: lambda char: ord(char) - 96,
    LabelStructureEnum.ROMAN_UC: lambda char: UC_ROMAN_TO_INT[char],
    LabelStructureEnum.ROMAN_LC: lambda char: LC_ROMAN_TO_INT[char]
}

class LabelStructure:
    """
    Class representing the label structure for a compound figure.
    With only two information (the type of labels and the number) it is possible
    to qualify the label structure of a figure.

    Attributes:
        labels_type (LabelStructureEnum):   The type of label structure.
        num_labels (int):                   The number of labels.
    """

    def __init__(self,
                 labels_type: LabelStructureEnum,
                 num_labels: int) -> None:
        """
        Args:
            labels_type (LabelStructureEnum):   The type of labels.
            num_labels (int):                   The number of labels.
        """

        self.labels_type: LabelStructureEnum = labels_type
        self.num_labels: int = num_labels


    @classmethod
    def from_labels_list(cls,
                         labels_list: List[str]) -> LabelStructure:
        """
        Create a LabelStructure object from a list of labels.

        Args:
            labels_list (List[str]):    A list of labels (text).

        Returns:
            label_structure (LabelStructure):   An instance of the corresponding LabelStructure
                                                    object.
        """
        # Remove duplicates
        labels_list = list(set(labels_list))
        # TODO maybe put in histogram...
        # Case where there are no named labels.
        if labels_list == ['_'] * len(labels_list):
            return cls(labels_type=LabelStructureEnum.NONE,
                       num_labels=len(labels_list))


        # "Histogram" of the label types.
        similarity_dict: Dict[LabelStructureEnum, float] = {
            structure: 0
            for structure in LabelStructureEnum}

        for label in labels_list:

            for label_type in (LabelStructureEnum.LATIN_LC,
                               LabelStructureEnum.LATIN_UC,
                               LabelStructureEnum.NUMERICAL,
                               LabelStructureEnum.ROMAN_LC,
                               LabelStructureEnum.ROMAN_UC):

                if LABEL_FILTER[label_type](label):
                    similarity_dict[label_type] += np.exp(-LABEL_INDEX[label_type](label))

        # TODO remove
        # pprint(similarity_dict)

        max_pos = max(similarity_dict.items(),
                      key=operator.itemgetter(1)
                      )[0]

        labels_type: LabelStructureEnum = LabelStructureEnum(max_pos)

        get_index: Callable[[str], int] = LABEL_INDEX[labels_type]

        # TODO Add comments to explain this ultra smart strategy

        # We start by removing duplicates.
        # ['C', '1', 'B', 'A', 'B'] -> ['C', '1', 'B', 'A']
        filtered_labels: List[str] = list(set(labels_list))

        # Keep only labels from the identified type:
        # ['C', '1', 'B', 'A'] -> ['C', 'B', 'A']
        filtered_labels = list(filter(LABEL_FILTER[labels_type],
                                      filtered_labels))
        # Sort the labels.
        # ['C', 'B', 'A'] -> ['A', 'B', 'C']
        filtered_labels.sort()


        if len(filtered_labels) > 1:

            # Keep labels until a 'gap' larger than 1:
            # ['A', 'B', 'D', 'G', 'H'] -> ['A', 'B', 'D']
            index: int = 1
            while index < len(filtered_labels)\
                and get_index(filtered_labels[index])\
                    - get_index(filtered_labels[index - 1]) <= 2:

                index += 1

            filtered_labels = filtered_labels[:index]

        # Reject a single label list that makes no sense (i.e. that doesn't approximately start
        # from the beginning).
        # ['P'] -> []
        if len(filtered_labels) == 1:
            if get_index(filtered_labels[0]) > 2:
                filtered_labels = []

        # Case where label list is empty
        if len(filtered_labels) == 0:
            return cls(labels_type=LabelStructureEnum.NONE,
                       num_labels=0)


        # Get the "last label"
        # ['A', 'B', 'C'] -> 'C'
        last_label: str = filtered_labels[-1]

        # Use the last label 'index' to infer the number of labels:
        # 'C' -> 3
        max_index: int = get_index(last_label)

        return cls(labels_type=labels_type,
                   num_labels=max_index)


    def get_core_label_list(self) -> List[str]:
        """
        Returns:
            core_label_list (List[str]):    The list of labels corresponding to this
                                                LabelStructure.
        """
        output_list: List[str] = []

        if self.labels_type is None:
            output_list = []

        elif self.labels_type == LabelStructureEnum.NUMERICAL:
            output_list = [str(i) for i in range(self.num_labels)]

        elif self.labels_type == LabelStructureEnum.LATIN_UC:
            output_list = [chr(65 + i) for i in range(self.num_labels)]

        elif self.labels_type == LabelStructureEnum.LATIN_LC:
            output_list = [chr(97 + i) for i in range(self.num_labels)]

        elif self.labels_type == LabelStructureEnum.ROMAN_UC:
            output_list = UC_ROMAN[:self.num_labels]

        elif self.labels_type == LabelStructureEnum.ROMAN_LC:
            output_list = LC_ROMAN[:self.num_labels]

        # Default case
        else:
            assert self.labels_type in (LabelStructureEnum.OTHER, LabelStructureEnum.NONE)
            output_list = ['_'] * self.num_labels

        return output_list

    def __eq__(self, other: object) -> bool:

        if isinstance(other, LabelStructure):
            return self.labels_type == other.labels_type and self.num_labels == other.num_labels

        return False

    def __str__(self) -> str:
        string: str = str(self.labels_type)
        string += f" | number of labels : {self.num_labels}"

        return string
