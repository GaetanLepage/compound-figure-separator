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


###################################################################
Functions to infer the labels from the caption text.

The original version of this code was written by Stefano Marchesin.
"""

from typing import List, Dict, Any, Tuple, cast
import re

from ..utils.figure.label import (is_char,
                                  is_roman,
                                  roman_to_int,
                                  LC_ROMAN_FROM_INT,
                                  UC_ROMAN_FROM_INT)

from . import regex_definitions


def label_identification(caption: str) -> Dict:
    """
    Given a caption text, identify the labels.

    Args:
        caption (str):  The caption text from which to extract captions.

    Returns:
        output_dict (Dict): A dict containing the detection information (list of labels, ranges).
    """
    # To avoid returning to many things, we build an output dict.
    output_dict: Dict[str, Any] = {
        'labels': {
            'characters': [],
            'romans': [],
            'digits': []
        },
        'ranges' : {
            'hyphen': None,
            'conj': None
        }
    }

    # Detect alphanumerical labels.
    characters_raw: List[str] = regex_definitions.RE_CHARACTERS.findall(caption)
    characters_cleaned: List[str] = []
    if characters_raw:
        # Get the list of alphanumerical labels.
        characters_list = []
        for raw in characters_raw:
            characters_list.append(raw[0])
        # Clean the list.
        for element in characters_list:
            characters_cleaned.append(re.sub(r'[().:]', '', element))

        # Remove duplicates.
        characters_cleaned = list(set(characters_cleaned))

        # Sort the list of characters.
        characters_cleaned.sort()

        # Store the list of labels in the output dict.
        output_dict['labels']['characters'] = characters_cleaned


    # Detect roman numbers.
    romans_raw: List[str] = regex_definitions.RE_ROMAN.findall(caption)
    romans_cleaned: List[str] = []
    if romans_raw:
        # Get the list of roman labels.
        romans_list: List[str] = [raw[0] for raw in romans_raw]

        # Clean the list.
        romans_cleaned = [re.sub(pattern=r'[().:]',
                                 repl='',
                                 string=element)
                          for element in romans_list]

        # Remove duplicates.
        romans_cleaned = list(set(romans_cleaned))

        # Sort the list of roman numbers according to their numerical values.
        romans_cleaned.sort(key=lambda roman_char: roman_to_int(roman_char=roman_char))

        # TODO remove
        # print(romans_cleaned)

        # Store the list of labels in the output dict.
        output_dict['labels']['romans'] = romans_cleaned


    # Detect numerical labels.
    digits_raw: List[str] = regex_definitions.RE_DIGITS.findall(caption)
    digits_cleaned: List[str] = []
    if digits_raw:
        # Get the list of numerical labels.
        digits_list = []
        for raw in digits_raw:
            digits_list.append(raw[0])

        # Clean the list.
        for element in digits_list:
            digits_cleaned.append(re.sub(pattern=r'[().:]',
                                         repl='',
                                         string=element))
        # remove duplicates
        digits_cleaned = list(set(digits_cleaned))
        # sort the list of characters
        digits_cleaned.sort()

        # Store the list of labels in the output dict.
        output_dict['labels']['digits'] = digits_cleaned


    # Get hyphens and conjunctions.
    hyphen_range = regex_definitions.RE_HYPHEN.findall(caption)
    conj_range = regex_definitions.RE_CONJUNCTIONS.findall(caption)

    # Extract first element of each tuple and replace the tuple with it.
    # Hyphen range.
    hyphen_vector: List[str] = [hyphen_tuple[0] for hyphen_tuple in hyphen_range]
    # Conjunction range.
    conj_vector: List[str] = [conj_tuple[0] for conj_tuple in conj_range]


    # Store the ranges in the output dict.
    output_dict['ranges']['hyphen'] = hyphen_vector
    output_dict['ranges']['conj'] = conj_vector

    return output_dict



def _expand_hyphen_range(hyphen_expressions: List[str]) -> List[str]:
    """
    Expand the label hyphen ranges from the caption.
    ex: ['A-C', 'E-F'] -> ['A', 'B', 'C', 'E', 'F']

    Args:
        hyphen_expressions (List[str]): The list of hyphen matches.

    Returns:
        hyphen_range (List[str]):   The list of expanded elements.
    """
    hyphen_range: List[str] = []

    for range_str in hyphen_expressions:

        # Split the string by hyphen: 'A-D' -> ['A', 'D']
        range_pair: Tuple[str, str] = cast(Tuple[str, str],
                                           range_str.split('-'))

        # Check if the range is numerical, roman or alphabetical.
        ## CAVEAT: set also the roman one
        # Case 1/3: numerical.
        if all(d.isdigit() for d in range_pair):

            # Get numerical lower and upper bounds.
            inf: int = int(range_pair[0])
            sup: int = int(range_pair[-1])

            hyphen_range += list(map(str,
                                     range(inf, sup + 1)))

        # Case 2/3: roman numbers.
        elif all(is_roman(r) for r in range_pair):

            # Check if roman numbers are lower or upper case.
            # Thanks to how regular expressions operate on '|' we just need to check first
            # element.
            is_upper = range_pair[0].isupper()

            # Get numerical lower and upper bounds.
            inf = roman_to_int(range_pair[0])
            sup = roman_to_int(range_pair[-1])

            # Expand the range of numerical numbers and revert it back to its roman form.
            roman_range_int: List[int] = list(range(inf, sup + 1))
            roman_range_str: List[str]

            if is_upper:
                roman_range_str = [UC_ROMAN_FROM_INT[value] for value in roman_range_int]
            else:
                roman_range_str = [LC_ROMAN_FROM_INT[value] for value in roman_range_int]

            # Concatenate the range of roman numbers to the list of ranges.
            hyphen_range += roman_range_str

        # Case 3/3: alphabetical characters.
        elif all(is_char(char=r) for r in range_pair):

            # Get 'numerical' lower and upper bounds.
            inf = ord(range_pair[0])
            sup = ord(range_pair[-1])
            hyphen_range += list(map(chr,
                                     range(inf, sup + 1)))

    return hyphen_range



def label_expansion(label_dict: Dict) -> List[str]:
    """
    Expand the label ranges from the caption.
    ex: ['A-C', 'D', 'E and F'] -> ['A', 'B', 'C', 'D', 'E', 'F']

    Args:
        label_dict (Dict):  The dict containing labels lists and ranges detections.

    Returns:
        labels (List[str]): The final list of detected labels.
    """
    ranges: List[str] = []

    # ==> Hyphen ranges.
    # Clean the elements and expand the sequences hyphen range.
    hyphen_cleaned: List[str] = [re.sub(pattern=r'[().:]',
                                        repl='',
                                        string=element)
                                 for element in label_dict['ranges']['hyphen']]

    ranges = _expand_hyphen_range(hyphen_expressions=hyphen_cleaned)


    # ==> Conjunction ranges.
    # Clean the identified patterns from useless characters.
    conj_range: List[str] = [re.sub(pattern=r'[().:,]',
                                    repl=' ',
                                    string=element.replace('and', ' '))
                             for element in label_dict['ranges']['conj']]

    # print('conj_cleaned:', conj_cleaned)
    # Append elements to ranges.
    for element in conj_range:
        ranges += element.split()
    # print("ranges before removing duplicates:", ranges)

    # Remove duplicates.
    ranges = list(set(ranges))

    # TODO remove
    # print("ranges after removing duplicates:", ranges)

    # Sort ranges.
    ranges.sort()
    # print("expanded ranges:", ranges)

    # Merge the lists containing the expanded ranges and the single labels.
    # ('|' is the union operation between sets).
    labels = list(set(ranges)\
                  | set(label_dict['labels']['digits'])\
                  | set(label_dict['labels']['romans'])\
                  | set(label_dict['labels']['characters'])
                  )

    # Sort the labels.
    labels.sort()

    return labels


# TODO check how to use this.
## 3rd step: label filtering
# Decide which list to consider (digits/romans or alphanumeric) depending on
# the amount of matched characters between image and caption.

### CAVEAT: This part will be done once we have the result list of characters from images ###

# CAVEAT: this list is the one that will be used once we know which are the correct labels.
# labels_final = []

# if (selected_labels==['none']):
    # selected_labels = labels_alphanum
