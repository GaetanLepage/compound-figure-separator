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


##################################################################
Functions to infer the labels from the caption text.

The original version of this code was written by Stefano Marchesin
(stefano.marchesin@unipd.it).
"""

from typing import List
import re

from ..utils.figure.labels_structure import LC_ROMAN, UC_ROMAN
from . import regex_definitions


def is_roman(string: str) -> bool:
    """
    Check if the given string is a roman number (up to 20).
    (i, ii,..., xx).

    Args:
        string (str):   A string that may be a roman number.

    Returns:
        is_roman (bool):    Whether the given string is a roman number.
    """
    return string in UC_ROMAN or string in LC_ROMAN


## 1st step: label identification
def label_identification(caption: str) -> List['str']:
    """
    TODO

    Args:
        caption (str):  The caption text from which to extract captions.

    Returns:
        label_list (List[str]): A list of detected labels.
    """
    print("\n## Step 1")
    # Detect alphanumerical labels.
    characters_raw = regex_definitions.RE_CHARACTERS.findall(caption)
    print(characters_raw)
    characters_cleaned = []
    if characters_raw:
        # Get the list of alphanumerical labels.
        characters_list = []
        for raw in characters_raw:
            characters_list.append(raw[0])
        # clean the list
        for element in characters_list:
            characters_cleaned.append(re.sub(r'[().:]', '', element))
        # remove duplicates
        characters_cleaned = list(set(characters_cleaned))
        # sort the list of characters
        characters_cleaned.sort()
        print("characters_cleaned:", characters_cleaned)

    # detect roman numbers
    romans_raw = regex_definitions.RE_ROMAN.findall(caption)
    romans_cleaned = []
    if romans_raw:
        # Get the list of roman labels.
        romans_list = [raw[0] for raw in romans_raw]

        print(romans_list)

        # Clean the list.
        romans_cleaned = [re.sub(pattern=r'[().:]',
                                 repl='',
                                 string=element)
                          for element in romans_list]


        # Remove duplicates.
        romans_cleaned = list(set(romans_cleaned))

        # Check if roman numbers are lower or upper case.
        # Thanks to how regular expressions operate on '|' we just need to check first element
        is_upper = romans_cleaned[0].isupper()
        # Convert roman to numerical.
        for key, value in enumerate(romans_cleaned):
            # Check if roman numbers are upper or lower case.
            if is_upper:
                romans_cleaned[key] = uc_latin_mapper[value]
            else:
                romans_cleaned[key] = lc_latin_mapper[value]
        # Sort the list of numerical numbers and revert it back to its roman
        # form.
        romans_cleaned.sort()
        for key, value in enumerate(romans_cleaned):
            # Check if roman numbers were upper or lower case.
            if is_upper:
                romans_cleaned[key] = uc_latin_mapper[value]
            else:
                romans_cleaned[key] = lc_latin_mapper[value]
        print("romans_cleaned:", romans_cleaned)

    # detect numerical labels
    digits_raw = regex_definitions.RE_DIGITS.findall(caption)
    digits_cleaned = []
    if digits_raw:
        # get the list of numerical labels
        digits_list = []
        for raw in digits_raw:
            digits_list.append(raw[0])
        # clean the list
        for element in digits_list:
            digits_cleaned.append(re.sub(r'[().:]', '', element))
        # remove duplicates
        digits_cleaned = list(set(digits_cleaned))
        # sort the list of characters
        digits_cleaned.sort()
        print("digits_cleaned:", digits_cleaned)


    # Get hyphens and conjunctions.
    hyphen_range = regex_definitions.RE_HYPHEN.findall(caption)
    conj_range = regex_definitions.RE_CONJUNCTIONS.findall(caption)

    print("hyphen_range:", hyphen_range)
    print("conj_range:", conj_range)

    # TODO
    return None


## 2nd step: label expansion
def label_expansion(caption,
                    detected_labels):
    """
    TODO
    """
    # Extract first element of each tuple and replace the tuple with it.
    ranges = []
    # Hyphen range.
    hyphen_vector = [hyphen_tuple[0] for hyphen_tuple in hyphen_range]
    # Conjunction range.
    conj_vector = [conj_tuple[0] for conj_tuple in conj_range]

    # Clean the elements and expand the sequences hyphen range.
    hyphen_cleaned = []
    for element in hyphen_vector:
        hyphen_cleaned.append(re.sub(pattern=r'[().:]',
                                     repl='',
                                     string=element))

    for element in hyphen_cleaned:

        # Split the string by hyphen
        element = element.split('-')

        # Check if the range is numerical, roman or alphabetical
        ## CAVEAT: set also the roman one
        if all(d.isdigit() for d in element): # numerical
            ranges = ranges + list(map(int,
                                       range(ord(element[0]),
                                             ord(element[-1]) + 1)))

        # Roman.
        elif all(is_roman(r) for r in element):

            # Convert roman numbers into numericals, expand and then re-convert.
            for key, value in enumerate(element):

                # Check if roman numbers are upper or lower case.
                if is_upper:
                    element[key] = uc_latin_mapper[value]
                else:
                    element[key] = lc_latin_mapper[value]

            # Expand the range of numerical numbers and revert it back to its roman form.
            roman_range = list(map(int,
                                   range(ord(element[0]),
                                         ord(element[-1]) + 1)))

            for key, value in enumerate(roman_range):
                # Check if roman numbers were upper or lower case.
                if is_upper:
                    roman_range[key] = uc_latin_mapper[value]
                else:
                    roman_range[key] = lc_latin_mapper[value]

            # Concatenate the range of roman numbers to the list of ranges
            ranges += roman_range

        # Alphabetical.
        else:
            ranges += list(map(chr,
                               range(ord(element[0]),
                                     ord(element[-1]) + 1)
                               )
                            )

    # Conjunction range.
    conj_cleaned = []

    # Clean the identified patterns from useless characters.
    for element in conj_vector:
        conj_cleaned.append(re.sub(pattern=r'[().:,]',
                                   repl=' ',
                                   string=element.replace('and', ' ')))
    # Append elements to ranges.
    for element in conj_cleaned:
        ranges += element.split()

    # Remove duplicates.
    ranges = list(set(ranges))

    ranges.sort()
    print("ranges:", ranges)


    # Merge the lists containing the expanded ranges and the single labels.
    # (union operation between sets).
    labels = list(set(ranges) | set(digits_cleaned) | set(romans_cleaned) | set(characters_cleaned))
    labels.sort()
    # Split the labels list into three sublists: digits, alphanumeric & roman.
    labels_digits = []
    labels_romans = []
    labels_alphanum = []

    for label in labels:

        # Store digit.
        if label.isdigit():
            labels_digits.append(label)

        # Store roman.
        elif is_roman(label):
            labels_romans.append(label)

        # Store alphanumerical.
        else:
            labels_alphanum.append(label)

    print("labels_digits:", labels_digits)
    print("labels_romans:", labels_romans)
    print("labels_alphanum:", labels_alphanum)


## 3rd step: label filtering
# Decide which list to consider (digits/romans or alphanumeric) depending on
# the amount of matched characters between image and caption.

### CAVEAT: This part will be done once we have the result list of characters from images ###

# CAVEAT: this list is the one that will be used once we know which are the correct labels.
# labels_final = []

# if (selected_labels==['none']):
    # selected_labels = labels_alphanum
