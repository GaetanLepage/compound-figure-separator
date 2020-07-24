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

Collaborators:  Niccolò Marini (niccolo.marini@hevs.ch)
                Stefano Marchesin (stefano.marchesin@unipd.it)


###############################
Defines a structure for labels.
"""

import logging
import operator
from enum import Enum
from typing import List

LC_ROMAN = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
            'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx']

# {'i': 1, 'ii': 2,..., 'xx': 20}
LC_ROMAN_TO_INT = {char: int_value
                   for char, int_value in zip(LC_ROMAN,
                                              range(1,
                                                    len(LC_ROMAN) + 1)
                                              )
                   }
# {1: 'i', 2: 'ii',..., 20: 'xx'}
LC_ROMAN_FROM_INT = {value: char for char, value in LC_ROMAN_TO_INT.items()}

# ['I', 'II',..., 'XX']
UC_ROMAN = [char.upper() for char in LC_ROMAN]

# {'I': 1, 'II': 2,..., 'XX': 20}
UC_ROMAN_TO_INT = {char: int_value
                   for char, int_value in zip(UC_ROMAN,
                                              range(1,
                                                    len(UC_ROMAN) + 1)
                                              )
                   }

# {1: 'I', 2: 'II',..., 20: 'XX'}
UC_ROMAN_FROM_INT = {value: char for char, value in UC_ROMAN_TO_INT.items()}




class LabelStructureEnum(Enum):
    """
    TODO
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
                 labels_type: LabelStructureEnum = None,
                 num_labels: int = None):
        """
        Args:
            labels_type (LabelStructureEnum):   The type of labels.
            num_labels (int):                   The number of labels.
        """

        self.labels_type = labels_type
        self.num_labels = num_labels


    @classmethod
    def from_labels_list(cls,
                         labels_list: List[str]) -> 'LabelStructure':
        """
        Create a LabelStructure object from a list of labels.

        Args:
            labels_list (List[str]):    A list of labels (text).

        Returns:
            label_structure (LabelStructure):   An instance of the corresponding LabelStructure
                                                    object.
        """
        # Case where there are no named labels.
        if labels_list == ['_'] * len(labels_list):
            return cls(labels_type=LabelStructureEnum.NONE,
                       num_labels=len(labels_list))


        # "Histogram" of the label types.
        similarity_list = {structure: 0 for structure in LabelStructureEnum}

        for label in labels_list:

            # Test if label is a latin character (single letter by definition).
            if len(label) == 1:

                # a, b, c...
                if ord(label) in range(97, 97 + 26):
                    similarity_list[LabelStructureEnum.LATIN_LC] += 1
                    continue

                # A, B, C...
                if ord(label) in range(65, 65 + 26):
                    similarity_list[LabelStructureEnum.LATIN_UC] += 1
                    continue

            # Test if label is an int.
            # 1, 2, 3...
            try:
                int(label)
                similarity_list[LabelStructureEnum.NUMERICAL] += 1
                continue
            except ValueError:
                pass

            # Test if label is a roman character (i, ii, iii...).
            # --> Can be several characters long.
            if label in LC_ROMAN_TO_INT.keys():
                similarity_list[LabelStructureEnum.ROMAN_LC] += 1
                continue

            if label in UC_ROMAN_TO_INT.keys():
                similarity_list[LabelStructureEnum.ROMAN_UC] += 1
                continue

            # Default case
            logging.warning(f"Label {label} does not belong to a default type.")
            similarity_list[LabelStructureEnum.OTHER] += 1

        max_pos = max(similarity_list.items(),
                      key=operator.itemgetter(1)
                      )[0]

        return cls(labels_type=LabelStructureEnum(max_pos),
                   num_labels=len(labels_list))


    def get_core_label_list(self) -> List[str]:
        """
        Returns:
            core_label_list (List[str]):    The list of labels corresponding to this
                                                LabelStructure.
        """
        if self.labels_type is None:
            return []

        if self.labels_type == LabelStructureEnum.NUMERICAL:
            list = [str(i) for i in range(self.num_labels)]

        if self.from_labels_list == LabelStructureEnum.LATIN_UC:
            return [chr(65 + i) for i in range(self.num_labels)]

        if self.from_labels_list == LabelStructureEnum.LATIN_LC:
            return [chr(97 + i) for i in range(self.num_labels)]

        if self.labels_type == LabelStructureEnum.ROMAN_UC:
            return UC_ROMAN[:self.num_labels]

        if self.labels_type == LabelStructureEnum.ROMAN_LC:
            return LC_ROMAN[:self.num_labels]

        if self.labels_type in (LabelStructureEnum.OTHER, LabelStructureEnum.NONE):
            return ['_'] * self.num_labels


    def __eq__(self, other: 'LabelStructure') -> bool:

        if isinstance(other, LabelStructure):
            return self.labels_type == other.labels_type and self.num_labels == other.num_labels

        return False

    def __str__(self) -> str:
        string = str(self.labels_type)
        string += f" | number of labels : {self.num_labels}"

        return string
