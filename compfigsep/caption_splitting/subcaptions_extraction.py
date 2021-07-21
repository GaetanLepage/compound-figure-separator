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


###################################################################
Functions for splitting a caption knowing the underlying labels.

The original version of this code was written by Stefano Marchesin.
"""

from __future__ import annotations
import re
from typing import NamedTuple, Optional
import itertools

import nltk

from .regex_definitions import (RE_DIGITS,
                                RE_DIGITS_POS,
                                RE_CHARACTERS,
                                RE_CHARACTERS_POS,
                                RE_ROMAN,
                                RE_ROMAN_POS,
                                RE_HYPHEN,
                                RE_CONJUNCTIONS,
                                RE_HYPHEN_POS,
                                RE_CONJUNCTIONS_POS)

from ..utils.figure.label import LabelStructure, LabelStructureEnum


OptionalRegexPair = dict[LabelStructureEnum,
                         tuple[Optional[re.Pattern],
                               Optional[re.Pattern]]]

LABEL_STRUCTURE_TO_REGEX: OptionalRegexPair = {
    LabelStructureEnum.NONE: (None, None),
    LabelStructureEnum.NUMERICAL: (RE_DIGITS, RE_DIGITS_POS),
    LabelStructureEnum.LATIN_UC: (RE_CHARACTERS, RE_CHARACTERS_POS),
    LabelStructureEnum.LATIN_LC: (RE_CHARACTERS, RE_CHARACTERS_POS),
    LabelStructureEnum.ROMAN_UC: (RE_ROMAN, RE_ROMAN_POS),
    LabelStructureEnum.ROMAN_LC: (RE_ROMAN, RE_ROMAN_POS),
}


class Position(NamedTuple):
    """
    A position is a tuple with the following format:
    """
    start_index: int
    end_index: int
    string_list: list[str]

    @classmethod
    def from_match(cls,
                   match: re.Match) -> Position:
        """
        Create a Position object from a regex Match object.

        Args:
            match (re.Match):   A Match object.

        Returns:
            position (Position):    The corresponding Position object.
        """
        return Position(start_index=match.start(),
                        end_index=match.end(),
                        string_list=[match.group(0)])


def _sentence_preface(sentences_list: list[str],
                      target_regex: re.Pattern) -> tuple[str, int]:
    """
    Search for the preface sentence in the caption.

    Args:
        sentences_list (list[str]): TODO.
        target_regex (re.Pattern):  Regular expression to detect labels.

    Returns:
        preface (str):  The preface sentence(s).
        index (int):    The index (in the splitted_sentence list) where the preface sentence ends.
    """
    # Define the preface string.
    preface: str = ''
    # Define preface_pos to set the delimiter for preface.
    preface_positions: list[Position] = []

    # Safe initialization as we return index at the end.
    index: int = 0

    # Check beginning sentences that do not contain any image pointer
    for index, sentence in enumerate(sentences_list):

        for regex in (target_regex,
                      RE_CONJUNCTIONS,
                      RE_HYPHEN):

            for match in regex.finditer(sentence):

                preface_positions.append(Position.from_match(match=match))

        # Stop as soon as a label is detected.
        if len(preface_positions) > 0:
            break

        # If preface_pos does not contain anything then append the sub sentence to preface.
        preface += sentence + ' '

    # TODO : should remove or not ?
    # If the script cannot find a preface, the first sentence is the preface.
    if preface == '':
        preface = sentences_list[0]
        index = 1

    return preface, index


def _label_positions(subcaption: str,
                     target_regex: re.Pattern) -> list[Position]:
    """
    Set the positions of labels within the sentence
    TODO

    Args:
        subcaption (str):           Sub-caption sentence.
        target_regex (re.Pattern):  Regular expression to detect labels.

    Returns:
        positions (list[Position]): list of the positions of detected labels.
    """
    # Loop through all the regex (i.e. char, hyphen and conj) and put them into
    # positions.
    positions: list[Position] = []

    # Conjunctions.
    for match in RE_CONJUNCTIONS.finditer(subcaption):
        # Expand the range into a list of image pointers.
        range_cleaned: str = re.sub(pattern=r'[().:,]',
                                    repl=' ',
                                    string=match.group(0).replace('and', ' '))

        # Only keep labels containing only alphanumerical characters.
        range_expnd: list[str] = [label
                                  for label in range_cleaned
                                  if label.isalnum()]

        # Create Position object and append it to the positions list.
        positions.append(Position(start_index=match.start(),
                                  end_index=match.end(),
                                  string_list=range_expnd))

    # Hyphen.
    for match in RE_HYPHEN.finditer(subcaption):
        range_expnd = []
        # Expand the range into a list of image pointers.
        range_cleaned = re.sub(pattern=r'[().:]',
                               repl='',
                               string=match.group(0))

        inf = ord(range_cleaned[0])
        sup = ord(range_cleaned[-1])
        label_range = range(inf, sup + 1)

        # Numerical range.
        if any(d.isdigit() for d in range_cleaned):
            range_expnd += list(map(chr,
                                    label_range))

        # Alphabetical range.
        else:
            range_expnd += list(map(chr,
                                    label_range))

        # Create Position object and append it to the positions list.
        positions.append(Position(start_index=match.start(),
                                  end_index=match.end(),
                                  string_list=range_expnd))

    # Target labels.
    for match in target_regex.finditer(subcaption):

        # Clean single labels from additional elements.
        char_cleaned = [re.sub(pattern=r'[().:,]',
                               repl='',
                               string=match.group(0))]

        positions.append(Position(start_index=match.start(),
                                  end_index=match.end(),
                                  string_list=char_cleaned))

    # TODO unclear how positions are sorted
    # see https://stackoverflow.com/a/5824559/11196710
    positions.sort()

    return positions


def _pos_positions(subcaption: str,
                   target_regex_pos: re.Pattern) -> list[Position]:
    """
    Set the positions of POS labels within the sentence.

    Args:
        subcaption (str):               The subcaption text.
        target_regex_POS (re.Pattern):  TODO.

    Returns:
        positions_POS (list[Position]): A list of tuple representing TODO.
    """
    # Loop through all the POS regex (i.e. target, hyphen and conj) and put them into
    # positions_POS.
    positions_pos: list[Position] = []

    # There is no need to expand the ranges as we are only interested in its position.
    for regex in (RE_CONJUNCTIONS_POS,
                  RE_HYPHEN_POS,
                  target_regex_pos):

        for match in regex.finditer(subcaption):

            positions_pos.append(Position.from_match(match=match))

    positions_pos.sort()

    return positions_pos


def _remove_overlaps(positions: list[Position]) -> list[Position]:
    """
    Remove overlapping elements (e.g. (A-D) and D)).

    Args:
        positions (list[Position]): A list of positions.

    Returns:
        positions (list[Position]): The updated list of positions.
    """
    mask: list[bool] = [True] * len(positions)

    for index, (position, next_position) in enumerate(zip(positions[:-1], positions[1:])):

        if position.end_index >= next_position.end_index:
            mask[index + 1] = False

    return list(itertools.compress(data=positions,
                                   selectors=mask))


def _remove_pos(positions: list[Position],
                positions_pos: list[Position]) -> None:
    """
    Remove from 'positions' elements that have been classified as POS labels.

    Args:
        positions (list[Position]):     A list of positions.
        positions_POS (list[Position]): A list of POS positions.
    """
    # Check for words that are associated to labels like in, from and panel within the sentence.
    # Check for elements within positions_POS that incorporate positions elements.
    for position_pos in positions_pos:

        # Check if the ending delimiter of an image pointer is equal to that of a POS label.
        for index, position in enumerate(positions):

            if position_pos.end_index == position.end_index:
                # If the two ending delimiters are equal then remove from positions the image
                # pointer (POS)
                positions.pop(index)


def _post_labels(subcaptions: dict[str, str],
                 subcapt: str,
                 positions: list[Position]) -> None:
    """
    Associate post description labels to sentences.

    Args:
        subcaptions (dict[str, str]):   dict of subcaptions to augment.
        subcapt (str):                  sub part of the caption to process.
        positions (list[Position]):     list of positions detected in `subcapt`.
    """
    # Deal with first position.
    end: int = positions[0].start_index

    # Avoid wrong cases like (A-D) ______ (B)____(C). Where (A-D) is clearly in an
    # incorrect position.
    if end != 0:
        # Loop through the list of labels attached to each position.
        label_list: list[str] = positions[0].string_list
        for label in label_list:
            # Inserted try catch for avoiding error when the script misleads (i,v) for
            # roman numbers.
            if label in subcaptions:
                subcaptions[label] += subcapt[:end] + '. '

    # Loop through all the other detected labels within the sentence.
    for index, position in enumerate(positions[1:]):

        prev_pos: Position = positions[index - 1]

        # Initial position equal to end delimiter of (index -1) + 1.
        init = prev_pos.end_index

        # Ending position equal to initial delimiter of index.
        end = position.start_index

        # Loop through the list of labels attached to each position.
        label_list = position.string_list
        for label in label_list:
            if label in subcaptions:
                subcaptions[label] += subcapt[init:end] + '. '


def _pre_labels(subcaptions: dict[str, str],
                subcapt: str,
                positions: list[Position]):
    """
    Associate pre description labels to sentences
    TODO

    Args:
        subcaptions (dict[str, str]):   dict of subcaptions to augment.
        subcapt (str):                  sub part of the caption to process.
        positions (list[Position]):     list of positions detected in `subcapt`.
    """
    # Loop through all the labels detected within the sentence but the last one.
    for position, next_position in zip(positions[:-1], positions[1:]):

        # Initial position equal to end delimiter of index + 1.
        init: int = position.end_index

        # Ending position equal to initial delimiter of (index + 1).
        end: int = next_position.start_index

        # If the caption is a single character, no point in adding it.
        if end - init <= 1:
            continue

        # Loop through the list of labels attached to each position.
        label_list: list[str] = position.string_list

        for label in label_list:
            subcaptions[label] += subcapt[init:end] + '. '


    # Deal with the last label.
    last_position = positions[-1]

    # `init` is where `last_position` ends
    init = last_position.end_index

    # Avoid wrong cases like (A) ____ (B) ____(C). Where (C) is clearly
    # in an incorrect position.
    if init != len(subcapt):
        # loop through the list of labels attached to each position
        label_list = last_position.string_list
        for label in label_list:
            if label in subcaptions:
                subcaptions[label] += subcapt[init:] + '. '


def _clean_positions(positions: list[Position],
                     selected_labels) -> list[Position]:
    """
    TODO

    Args:
        positions (list[Position]):     TODO.
        selected_labels (list[str]):    TODO.

    Returns:
        new_positions (list[Position]): TODO.
    """
    new_positions = []
    for position in positions:
        list_labels_expanded = position.string_list

        if all(label in selected_labels
               for label in list_labels_expanded):

            new_positions.append(position)

    return new_positions


def _process_caption_subsentence(caption_subsentence: str,
                                 sub_labels: list[str],
                                 sub_image_pointers: list[str],
                                 subcaptions: dict[str, str],
                                 fuzzy_captions: dict[str, str],
                                 target_regex: re.Pattern) -> None:
    """
    TODO

    Args:
        caption_subsentence (str):          TODO.
        sub_labels (list[str]):             TODO.
        subcaptions (dict[str, str]):       TODO.
        fuzzy_captions (dict[str, str]):    TODO.
        target_regex (re.Pattern):          Regular expression to detect labels.
    """

    # For each subsentence extract all the image pointers and their positions.

    # Define the list of tuples representing image pointers
    sub_positions: list[Position] = []

    # Loop through all the regex (i.e. char, hyphen and conj) and put them
    # into sub_positions.
    sub_positions = _label_positions(subcaption=caption_subsentence,
                                     target_regex=target_regex)

    # Remove overlapping elements (e.g. (A-D) and D)).
    sub_positions = _remove_overlaps(positions=sub_positions)

    # Compute the length of each subsentence.
    subsentence_len = len(caption_subsentence)

    # Check if sub_positions list is empty or not.
    if sub_positions:
        # Assign to sub_image_pointers the list of labels for the subsentence.
        # TODO rephrase:
        # (It will be kept until a new subsentence with labels won't be found).
        temp: list[str] = []
        for sub_position in sub_positions:
            temp += sub_position.string_list

        sub_image_pointers = list(set(temp))

        # Classify labels in 'pre', 'post' and 'in' description (remember that
        # there are no POS in this case).
        # Check if the last label extracted is at the end of the subsentence.
        if sub_positions[-1].end_index == subsentence_len:
            # Consider labels as post description labels.

            # Assign to the subcaptions the related sentences
            _post_labels(subcaptions=subcaptions,
                         subcapt=caption_subsentence,
                         positions=sub_positions)

        # Check if the first label extracted is at the beginning of the
        # subsentence.
        elif sub_positions[0].start_index == 0:
            # Consider labels as 'pre description' labels.

            # Assign to the subcaptions the related subsentences.
            _pre_labels(subcaptions=subcaptions,
                        subcapt=caption_subsentence,
                        positions=sub_positions)

        # Consider labels as 'fuzzy labels' and store the subsentence in
        # fuzzycaptions.
        else:
            for label in sub_labels:
                fuzzy_captions[label] += caption_subsentence + ' '

    # Case where sub_positions is empty.
    # Assign the subsentence without labels to the subcaptions that have been
    # expanded in the previous iteration.
    else:
        for label in sub_image_pointers:
            subcaptions[label] += caption_subsentence


def _process_caption_subsentence_pos(caption_subsentence,
                                     sub_labels: list[str],
                                     sub_image_pointers: list[str],
                                     subcaptions: dict[str, str],
                                     fuzzy_captions: dict[str, str],
                                     target_regex: re.Pattern,
                                     target_regex_pos: re.Pattern) -> None:
    """
    TODO

    Args:
        caption_subsentence (str):          TODO.
        sub_labels (list[str]):             TODO.
        sub_image_pointers (list[str]):     TODO.
        subcaptions (dict[str, str]):       TODO.
        fuzzy_captions (dict[str, str]):    TODO.
        target_regex (re.Pattern):          TODO.
        target_regex_pos (re.Pattern):      TODO.
    """
    # For each subsentence extract all the image pointers and their positions.

    # Define the list of tuples representing image pointers.
    sub_positions: list[Position] = []

    # Define the list of tuples representing POS labels.
    sub_positions_pos: list[Position] = []

    # Loop through all the regex (i.e. char, hyphen and conj) and put them
    # into sub_positions.
    sub_positions = _label_positions(subcaption=caption_subsentence,
                                     target_regex=target_regex)

    # Loop through all the POS regex (i.e. char, hyphen and conj) and put
    # them into sub_positions_POS.
    sub_positions_pos = _pos_positions(subcaption=caption_subsentence,
                                       target_regex_pos=target_regex_pos)

    # Remove overlapping elements (e.g. (A-D) and D)).
    sub_positions = _remove_overlaps(positions=sub_positions)
    # Compute the length of each subsentence.
    subsentence_len = len(caption_subsentence)

    # Check if sub_positions list is empty or not.
    if sub_positions:

        # Assign to sub_image_pointers the list of labels for the subsentence
        # (it will be kept until a new subsentence with labels won't be
        # found).
        temp: list[str] = []
        for sub_position in sub_positions:
            temp += sub_position.string_list

        sub_image_pointers = list(set(temp))

        # Classify labels in pre, post and in description

        # Check if sub_positions_POS is empty or not.
        if sub_positions_pos:
            # Check if the last label extracted is at the end of the
            # subsentence and it is not contained in sub_positions_POS.
            if sub_positions[-1].end_index == subsentence_len\
                and sub_positions[-1].end_index != sub_positions_pos[-1].end_index:
                # Consider labels as 'post description' labels.

                # Check for words that are associated to labels like in, from
                # and panel within the subsentence.
                _remove_pos(positions=sub_positions,
                            positions_pos=sub_positions_pos)

                # Assign to the subcaptions the related subsentences.
                _post_labels(subcaptions=subcaptions,
                             subcapt=caption_subsentence,
                             positions=sub_positions)

            # Check if the first label extracted is at the beginning of the
            # sentence.
            elif sub_positions[0].start_index == 0:
                # Consider labels as 'pre description' labels.

                # Check for words that are associated to labels like in, from
                # and panel within the subsentence.
                _remove_pos(positions=sub_positions,
                            positions_pos=sub_positions_pos)

                # Assign to the subcaptions the related sentences.
                _pre_labels(subcaptions=subcaptions,
                            subcapt=caption_subsentence,
                            positions=sub_positions)

            # Consider labels as 'fuzzy labels' and store the subsentence in
            # fuzzycaptions.
            else:
                for label in sub_labels:
                    fuzzy_captions[label] += caption_subsentence + ' '

        # Case where `sub_positions_POS` is empty.
        else:
            # Check if the last label extracted is at the end of the
            # subsentence.
            if sub_positions[-1].end_index == subsentence_len:
                # Consider labels as 'post description' labels.

                # Assign to the subcaptions the related sentences.
                _post_labels(subcaptions=subcaptions,
                             subcapt=caption_subsentence,
                             positions=sub_positions)

            # Check if the first label extracted is at the beginning of the
            # subsentence.
            elif sub_positions[0].start_index == 0:
                # Consider labels as 'pre description' labels.

                # Assign to the subcaptions the related subsentences.
                _pre_labels(subcaptions=subcaptions,
                            subcapt=caption_subsentence,
                            positions=sub_positions)

            # Consider labels as 'fuzzy labels' and store the subsentence in
            # fuzzycaptions.
            else:
                for label in sub_labels:
                    fuzzy_captions[label] += caption_subsentence + ' '

    # Case where `sub_positions` is empty.
    # Assign the subsentence without labels to the subcaptions that have been
    # expanded in the previous iteration.
    else:
        for label in sub_image_pointers:
            subcaptions[label] += caption_subsentence



def _process_caption_sentence(caption_sentence: str,
                              subcaptions: dict[str, str],
                              fuzzy_captions: dict[str, str],
                              image_pointers: list[str],
                              filtered_labels: list[str],
                              target_regex: re.Pattern,
                              target_regex_pos: re.Pattern) -> list[str]:
    """
    TODO

    Args:
        caption_sentence (str):             The caption sentence to be processed.
        subcaptions (dict[str, str]):       The dictionary for outputing subcaptions.
        fuzzy_captions (dict[str, str]):    TODO.
        image_pointers (list[str]):         TODO.
        filtered_labels (list[str]):        TODO.
        target_regex (re.Pattern):          TODO.
        target_regex_pos (re.Pattern):      TODO.

    Returns:
        image_pointers (list[str]):         The latent list of image pointers.
        # TODO: do the same for subsentence and subsentence_pos.
    """

    # Define the list of tuples representing image pointers.
    # Loop through all the regex (i.e. target, hyphen and conj) and put them into positions.
    positions: list[Position] = _label_positions(subcaption=caption_sentence,
                                                 target_regex=target_regex)

    # remove overlapping elements (e.g. (A-D) and D))
    positions = _remove_overlaps(positions=positions)

    positions = _clean_positions(positions=positions,
                                 selected_labels=filtered_labels)

    # Check if positions list is empty or not.
    if len(positions) == 0:

        # In this case, assign the sentence without labels to the subcaptions that have been
        # expanded in the previous iteration.
        for label in image_pointers:
            subcaptions[label] += caption_sentence

        # Go to next sentence.
        return image_pointers

    # Assign to image_pointers the list of labels for the sentence (it will be kept until
    # a new sentence with labels won't be found).
    image_pointers = []
    for position in positions:
        image_pointers += position.string_list
    # Remove duplicates.
    image_pointers = list(set(image_pointers))

    # Classify labels in pre, post and in description.

    # Define the list of tuples representing POS labels.
    # Loop through all the POS regex (i.e. target, hyphen and conj) and put them into
    # positions_POS.
    positions_pos: list[Position] = _pos_positions(subcaption=caption_sentence,
                                                   target_regex_pos=target_regex_pos)

    # Define the list of image pointers that each subsentence contains.
    sub_image_pointers: list[str] = []

    # Check if positions_POS is empty or not.
    if positions_pos:

        # Check if the last label extracted is at the end of the sentence and it is not
        # contained in positions_pos.
        if positions[-1].end_index == len(caption_sentence) \
            and positions[-1].end_index != positions_pos[-1].end_index:
            # Consider labels as 'post description' labels.

            # Check for words that are associated to labels like 'in', 'from' and 'panel'
            # within the sentence.
            _remove_pos(positions=positions,
                        positions_pos=positions_pos)

            # Assign to the subcaptions the related sentences.
            _post_labels(subcaptions=subcaptions,
                         subcapt=caption_sentence,
                         positions=positions)

        # Check if the first extracted label is at the beginning of the sentence.
        elif positions[0].start_index == 0:
            # Consider labels as pre description labels.

            # Check for words that are associated to labels like 'in', 'from' and 'panel'
            # within the sentence.
            _remove_pos(positions=positions,
                        positions_pos=positions_pos)

            # Assign to the subcaptions the related sentences.
            _pre_labels(subcaptions=subcaptions,
                        subcapt=caption_sentence,
                        positions=positions)

        # Consider labels as 'in descriptions' labels.
        else:
            # Split the sentence according to ';'.
            splitted_sentence = re.split(pattern=';',
                                         string=caption_sentence)

            # add ; to each element but the last
            for subsentence in splitted_sentence[:-1]:
                subsentence += ';'

            # Obtain all the labels of the sentence from positions.
            sub_labels: list[str] = []
            for position in positions:
                # Concatenate the different lists contained in positions.
                sub_labels += position.string_list

            # Remove unnecessary duplicates.
            sub_labels = list(set(sub_labels))

            # Sort the labels.
            sub_labels.sort()

            # Obtain the preface string and the counter to use for starting to consider
            # non-preface sentences.
            sub_preface, sub_preface_end_index = _sentence_preface(
                sentences_list=splitted_sentence,
                target_regex=target_regex)

            # If preface is not an empty string then the preface has to be associated to
            # subpanel captions within sub_labels.
            if sub_preface != '':
                # Associate preface to each subcaption contained in sub_labels.
                for label in sub_labels:
                    subcaptions[label] += sub_preface

            # Loop through all the subsentences of the sentence after preface.
            for caption_subsentence in splitted_sentence[sub_preface_end_index:]:

                _process_caption_subsentence_pos(caption_subsentence=caption_subsentence,
                                                 sub_labels=sub_labels,
                                                 sub_image_pointers=sub_image_pointers,
                                                 subcaptions=subcaptions,
                                                 fuzzy_captions=fuzzy_captions,
                                                 target_regex=target_regex,
                                                 target_regex_pos=target_regex_pos)

    # Case where positions_pos is empty.
    else:

        # print("positions_pos is empty.")
        # Check if the last extracted label is at the end of the sentence.
        if positions[-1].end_index == len(caption_sentence):
            # Consider labels as 'post description' labels.

            # Assign to the subcaptions the related sentences.
            _post_labels(subcaptions=subcaptions,
                         subcapt=caption_sentence,
                         positions=positions)

        # Check if the first label extracted is at the beginning of the sentence.
        elif positions[0].start_index == 0:
            # Consider labels as 'pre description' labels.

            # Assign to the subcaptions the related sentences.
            _pre_labels(subcaptions=subcaptions,
                        subcapt=caption_sentence,
                        positions=positions)

        # Consider labels as 'in descriptions' labels.
        else:
            # Split the sentence according to ;
            splitted_sentence = re.split(pattern=';',
                                         string=caption_sentence)

            # Add ; to each element but the last.
            for subsentence in splitted_sentence[:-1]:
                subsentence += ';'

            # Obtain all the labels of the sentence from positions.
            sub_labels = []

            for position in positions:
                # concatenate the different lists contained in positions
                sub_labels += position.string_list

            # Remove unnecessary duplicates.
            sub_labels = list(set(sub_labels))

            # Sort the labels.
            sub_labels.sort()

            # Obtain the preface string and the counter to use for starting to consider
            # non-preface sentences.
            sub_preface, sub_preface_end_index = _sentence_preface(
                sentences_list=splitted_sentence,
                target_regex=target_regex)

            # If preface is not an empty string then the preface has to be associated to
            # subpanel captions within sub_labels.
            if sub_preface != '':
                # Associate preface to each subcaption contained in sub_labels.
                for label in sub_labels:
                    subcaptions[label] += sub_preface

            # Set the starter point for looping through the subsentences to counter.
            sub_starter = sub_preface_end_index


            # Loop through all the subsentences of the sentence after preface.
            for caption_subsentence in splitted_sentence[sub_starter:]:

                _process_caption_subsentence(caption_subsentence=caption_subsentence,
                                             sub_labels=sub_labels,
                                             sub_image_pointers=sub_image_pointers,
                                             subcaptions=subcaptions,
                                             fuzzy_captions=fuzzy_captions,
                                             target_regex=target_regex)

    return image_pointers


def extract_subcaptions(caption: str,
                        label_structure: LabelStructure) -> dict[str, str]:
    """
    Split the caption in sentences.

    Args:
        caption (str):                      The caption to split.
        label_structure (LabelStructure):   The infered label structure for this figure.
                                                This is isomorphic to giving a list of labels.

    Returns:
        subcaptions (dict[str, str]):   The dictionary containing the detected subcaptions.
    """
    # Case where the figure is not a compound figure. The caption does not need to be split.
    if label_structure.num_labels == 0:
        return {'_': caption}

    # Turn the label structure into a list of labels.
    filtered_labels: list[str] = label_structure.get_core_label_list()

    # Initialize the dictionaries containing the subcaptions.
    subcaptions: dict[str, str] = {label: '' for label in filtered_labels}
    fuzzy_captions: dict[str, str] = {label: '' for label in filtered_labels}

    # Get the regex from the label_structure.
    target_regex, target_regex_pos = LABEL_STRUCTURE_TO_REGEX[label_structure.labels_type]

    if target_regex is None and target_regex_pos is None:
        return {'_': caption}

    # Split the caption to a list of sentences.
    caption_sentences_list = nltk.sent_tokenize(caption)

    # Obtain the preface string and the counter to use for starting to consider
    # non-preface sentences.
    preface, preface_end_index = _sentence_preface(sentences_list=caption_sentences_list,
                                                   target_regex=target_regex)

    # CAVEAT: remember to check whether or not the preface (when it's not empty) matches one of
    # the image pointers extracted previously, otherwise remove false positives from it.

    # If preface is not an empty string then the preface has to be associated to each
    # subpanel caption.
    if preface != '':

        # Associate preface to each subcaption.
        for label in subcaptions:
            subcaptions[label] += preface

    # Define the list of image pointers that each sentence contains.
    image_pointers: list[str] = []

    # Loop through all the sentences of the caption after preface:
    for caption_sentence in caption_sentences_list[preface_end_index:]:
        # For each substring extract all the image pointers (labels) and their positions.

        image_pointers = _process_caption_sentence(caption_sentence=caption_sentence,
                                                   subcaptions=subcaptions,
                                                   fuzzy_captions=fuzzy_captions,
                                                   image_pointers=image_pointers,
                                                   filtered_labels=filtered_labels,
                                                   target_regex=target_regex,
                                                   target_regex_pos=target_regex_pos)

    return subcaptions
