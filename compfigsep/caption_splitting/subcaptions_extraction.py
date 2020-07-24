#!/usr/bin/env python3

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
Functions for splitting a caption knowing the underlying labels.

The original version of this code was written by Stefano Marchesin.
"""

from __future__ import annotations
import re
from typing import List, Dict, Tuple, NamedTuple

import nltk # type: ignore

from .regex_definitions import *


# TODO maybe rename into "Match" later.
class Position(NamedTuple):
    """
    A position is a tuple with the following format:
    """
    start_index: int
    end_index: int
    string_list: List[str]

    @classmethod
    def from_match(cls,
                   match: re.Match
                   ) -> Position:
        """
        TODO

        Args:
            match (re.Match):   TODO.
        """

        return Position(start_index=match.start(),
                        end_index=match.end(),
                        string_list=[match.group(0)])


def _sentence_preface(sentences_list: List[str],
                      target_regex: re.Pattern) -> Tuple[str, int]:
    """
    Search for the preface sentence in the caption.

    Args:
        sentences_list (List[str]): TODO.
        target_regex (re.Pattern):  TODO.

    Returns:
        preface (str):  TODO.
        index (int):    The index (in the splitted_sentence list) where the preface sentence ends.
    """
    # Define the preface string.
    preface: str = ''
    # Define preface_pos to set the delimiter for preface.
    preface_pos: List[Position] = []

    # Safe initialization as we return index at the end.
    index: int = 0
    # Check beginning sentences that do not contain any image pointer
    for index, sentence in enumerate(sentences_list):

        for regex in (target_regex,
                      RE_CONJUNCTIONS,
                      RE_HYPHEN):

            for match in regex.finditer(sentence):

                preface_pos.append(Position.from_match(match=match))

        # Stop as soon as a label is detected.
        if len(preface_pos) > 0:
            break

        # If preface_pos does not contain anything then append the sub sentence to preface.
        preface += sentence + ' '

    # If the script cannot find a preface, the first sentence is the preface.
    if preface == '':
        preface = sentences_list[0]

    return preface, index


def _label_positions(subcaption: str,
                     target_regex: re.Pattern
                     ) -> List[Position]:
    """
    Set the positions of labels within the sentence
    TODO

    Args:
        subcaption (str):           TODO.
        target_regex (re.Pattern):  TODO.

    Returns:
        positions (List[Position]): TODO.
    """
    # Loop through all the regex (i.e. char, hyphen and conj) and put them into
    # positions.
    positions: List[Position] = []

    # Conjunctions.
    for match in RE_CONJUNCTIONS.finditer(subcaption):
        # Expand the range into a list of image pointers.
        range_cleaned = re.sub(pattern=r'[().:,]',
                               repl=' ',
                               string_list=[match.group(0).replace('and', ' ')])

        # Only keep labels containing only alphanumerical characters.
        range_expnd: List[str] = [label
                                  for label in range_cleaned
                                  if label.isalnum()]

        positions.append(Position(start_index=match.start(),
                                  end_index=match.end(),
                                  string_list=range_expnd))

    # Hyphen.
    for match in RE_HYPHEN.finditer(subcaption):
        range_expnd = []
        # Expand the range into a list of image pointers.
        range_cleaned = re.sub(pattern=r'[().:]',
                               repl='',
                               string_list=[match.group(0)])

        inf = ord(range_cleaned[0])
        sup = ord(range_cleaned[-1])
        label_range = range(inf, sup + 1)

        # Numerical range.
        if any(d.isdigit() for d in range_cleaned):
            range_expnd += list(map(int,
                                    label_range))

        # Alphabetical range.
        else:
            range_expnd += list(map(chr,
                                    label_range))

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
                   target_regex_POS: re.Pattern = RE_CHARACTERS_POS
                   ) -> List[Position]:
    """
    Set the positions of POS labels within the sentence.

    Args:
        subcaption (str):               The subcaption text.
        target_regex_POS (re.Pattern):  TODO.

    Returns:
        positions_POS (List[Position]): A list of tuple representing TODO.
    """
    # Loop through all the POS regex (i.e. target, hyphen and conj) and put them into
    # positions_POS.
    positions_pos: List[Position] = []

    # There is no need to expand the ranges as we are only interested in its position.
    for regex in (RE_CONJUNCTIONS_POS,
                  RE_HYPHEN_POS,
                  target_regex_POS):

        for match in regex.finditer(subcaption):

            positions_pos.append(Position.from_match(match=match))

    positions_pos.sort()

    return positions_pos


def _remove_overlaps(positions: List[Position]) -> None:
    """
    Remove overlapping elements (e.g. (A-D) and D)).

    Args:
        positions (List[Position]): A list of positions.
    """
    # Remove overlapping elements (e.g. (A-D) and D)).
    for index, position in enumerate(positions[:-1]):

        next_pos = positions[index + 1]

        # Check if the end delimiter is greater than or equal to the following one.
        if position.end_index >= next_pos.end_index:

            # If true then remove the index + 1 element since it is redundant.
            positions.pop(index + 1)


def _remove_pos(positions: List[Position],
                positions_pos: List[Position]) -> None:
    """
    Remove from 'positions' elements that have been classified as POS labels.

    Args:
        positions (List[Position]):     A list of positions.
        positions_POS (List[Position]): A list of POS positions.
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


def _post_labels(subcaptions: Dict[str, str],
                 subcapt: str,
                 positions: List[Position]):
    """
    Associate post description labels to sentences.

    Args:
        subcaptions (Dict[str, str]):   Dict of subcaptions to augment.
        subcapt (str):                  sub part of the caption to process.
        positions (List[Position]):     List of positions detected in `subcapt`.
    """
    # Deal with first position.
    end = positions[0].start_index

    # Avoid wrong cases like (A-D) ______ (B)____(C). Where (A-D) is clearly in an
    # incorrect position.
    if end != 0:
        # Loop through the list of labels attached to each position.
        label_list = positions[0].string_list
        for label in label_list:
            # Inserted try catch for avoiding error when the script misleads (i,v) for
            # roman numbers.
            try:
                subcaptions[label] += subcapt[:end] + '. '
            except:
                # TODO
                pass

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
            try:
                subcaptions[label] += subcapt[init:end] + '. '
            except:
                # TODO
                pass


def _pre_labels(subcaptions: Dict[str, str],
                subcapt: str,
                positions: List[Position]):
    """
    Associate pre description labels to sentences
    TODO

    Args:
        subcaptions (Dict[str, str]):   Dict of subcaptions to augment.
        subcapt (str):                  sub part of the caption to process.
        positions (List[Position]):     List of positions detected in `subcapt`.
    """
    # Loop through all the labels detected within the sentence but the last one.
    for position_index, position in enumerate(positions[:-1]):

        next_position: Position = positions[position_index + 1]

        # Initial position equal to end delimiter of index + 1.
        init: int = position.end_index

        # Ending position equal to initial delimiter of (index + 1).
        end: int = next_position.start_index

        # Loop through the list of labels attached to each position.
        label_list: List[str] = position.string_list

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
            subcaptions[label] += subcapt[init:] + '. '


def _clean_positions(positions: List[Position],
                     selected_labels) -> List[Position]:
    """
    TODO

    Args:
        positions (List[Position]):     TODO.
        selected_labels (List[str]):    TODO.

    Returns:
        new_positions (List[Position]): TODO.
    """
    new_positions = []
    for position in positions:
        list_labels_expanded = position.string_list

        if all(label in selected_labels
               for label in list_labels_expanded):

            new_positions.append(position)

    return new_positions


def _process_caption_subsentence(caption_subsentence: str,
                                 sub_labels: List[str],
                                 sub_image_pointers: List[str],
                                 subcaptions: Dict[str, str],
                                 fuzzy_captions: Dict[str, str],
                                 target_regex: re.Pattern) -> None:
    """
    TODO

    Args:
        caption_subsentence (str):          TODO.
        sub_labels (List[str]):             TODO.
        subcaptions (Dict[str, str]):       TODO.
        fuzzy_captions (Dict[str, str]):    TODO.
        target_regex (re.Pattern):          TODO.
    """

    # For each subsentence extract all the image pointers and their positions.

    # Define the list of tuples representing image pointers
    sub_positions: List[Position] = []

    # Loop through all the regex (i.e. char, hyphen and conj) and put them
    # into sub_positions.
    sub_positions = _label_positions(subcaption=caption_subsentence,
                                     target_regex=target_regex)
    print("sub_positions:", sub_positions)

    # Remove overlapping elements (e.g. (A-D) and D)).
    _remove_overlaps(positions=sub_positions)

    # Compute the length of each subsentence.
    subsentence_len = len(caption_subsentence)

    # Check if sub_positions list is empty or not.
    if sub_positions:
        # Assign to sub_image_pointers the list of labels for the subsentence.
        # TODO rephrase:
        # (It will be kept until a new subsentence with labels won't be found).
        temp: List[str] = []
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
                                     sub_labels: List[str],
                                     sub_image_pointers: List[str],
                                     subcaptions: Dict[str, str],
                                     fuzzy_captions: Dict[str, str],
                                     target_regex: re.Pattern,
                                     target_regex_POS: re.Pattern) -> None:
    """
    TODO
    """
    # For each subsentence extract all the image pointers and their positions.

    # Define the list of tuples representing image pointers.
    sub_positions: List[Position] = []

    # Define the list of tuples representing POS labels.
    sub_positions_pos: List[Position] = []

    # Loop through all the regex (i.e. char, hyphen and conj) and put them
    # into sub_positions.
    sub_positions = _label_positions(subcaption=caption_subsentence,
                                     target_regex=target_regex)

    # Loop through all the POS regex (i.e. char, hyphen and conj) and put
    # them into sub_positions_POS.
    sub_positions_pos = _pos_positions(subcaption=caption_subsentence,
                                       target_regex_POS=target_regex_POS)

    # Remove overlapping elements (e.g. (A-D) and D)).
    _remove_overlaps(positions=sub_positions)
    # Compute the length of each subsentence.
    subsentence_len = len(caption_subsentence)

    # Check if sub_positions list is empty or not.
    if sub_positions:

        # Assign to sub_image_pointers the list of labels for the subsentence
        # (it will be kept until a new subsentence with labels won't be
        # found).
        temp: List[str] = []
        for sub_position in sub_positions:
            temp += sub_position.string_list

        sub_image_pointers = list(set(temp))

        # Classify labels in pre, post and in description

        # Check if sub_positions_POS is empty or not.
        if sub_positions_pos:
            # Check if the last label extracted is at the end of the
            # subsentence and it is not contained in sub_positions_POS.
            if sub_positions[-1][1] == subsentence_len\
                    and sub_positions[-1][1] != sub_positions_pos[-1][1]:
                # Consider labels as 'post description' labels.

                # Check for words that are associated to labels like in, from
                # and panel within the subsentence.
                _remove_pos(sub_positions, sub_positions_pos)

                # Assign to the subcaptions the related subsentences.
                _post_labels(subcaptions=subcaptions,
                             subcapt=caption_subsentence,
                             positions=sub_positions)

            # Check if the first label extracted is at the beginning of the
            # sentence.
            elif sub_positions[0][0] == 0:
                # Consider labels as 'pre description' labels.

                # Check for words that are associated to labels like in, from
                # and panel within the subsentence.
                _remove_pos(sub_positions, sub_positions_pos)

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
            if sub_positions[-1][1] == subsentence_len:
                # Consider labels as 'post description' labels.

                # Assign to the subcaptions the related sentences.
                _post_labels(subcaptions=subcaptions,
                             subcapt=caption_subsentence,
                             positions=sub_positions)

            # Check if the first label extracted is at the beginning of the
            # subsentence.
            elif sub_positions[0][0] == 0:
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
                              subcaptions: Dict[str, str],
                              fuzzy_captions: Dict[str, str],
                              image_pointers: List[str],
                              filtered_labels: List[str],
                              target_regex: re.Pattern,
                              target_regex_pos: re.Pattern) -> None:
    """
    TODO

    Args:
        caption_sentence (str):             TODO.
        subcaptions (Dict[str, str]):       The dictionary for outputing subcaptions.
        fuzzy_captions (Dict[str, str]):    TODO.
        image_pointers (List[str]):         TODO.
        filtered_labels (List[str]):        TODO.
        target_regex (re.Pattern):          TODO.
        target_regex_pos (re.Pattern):      TODO.
    """

    # Define the list of tuples representing image pointers.
    # Loop through all the regex (i.e. target, hyphen and conj) and put them into positions.
    positions: List[Position] = _label_positions(subcaption=caption_sentence,
                                                 target_regex=target_regex)
    print("positions :", positions)

    # remove overlapping elements (e.g. (A-D) and D))
    _remove_overlaps(positions=positions)
    print("positions (without overlaps) :", positions)

    positions = _clean_positions(positions=positions,
                                 selected_labels=filtered_labels)
    print("positions (cleaned) :", positions)

    # Check if positions list is empty or not.
    if len(positions) == 0:

        # In this case, assign the sentence without labels to the subcaptions that have been
        # expanded in the previous iteration.
        for label in image_pointers:
            subcaptions[label] += caption_sentence

        # Go to next sentence.
        return

    # Assign to image_pointers the list of labels for the sentence (it will be kept until
    # a new sentence with labels won't be found).
    image_pointers = []
    for position in positions:
        image_pointers += position.string_list
    # Remove duplicates.
    image_pointers = list(set(image_pointers))

    # Classify labels in pre, post and in description.
    print("image_pointers :", image_pointers)

    # Define the list of tuples representing POS labels.
    # Loop through all the POS regex (i.e. target, hyphen and conj) and put them into
    # positions_POS.
    positions_pos: List[Position] = _pos_positions(subcaption=caption_sentence,
                                                   target_regex_POS=target_regex_pos)
    print("positions_POS :", positions_pos)

    # Define the list of image pointers that each subsentence contains.
    sub_image_pointers: List[str] = []

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
            sub_labels: List[str] = []
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
                                                 target_regex_POS=target_regex_pos)

    # Case where positions_pos is empty.
    else:

        print("positions_pos is empty.")
        # Check if the last extracted label is at the end of the sentence.
        if positions[-1].end_index == len(caption_sentence):
            # Consider labels as 'post description' labels.

            print("last extracted label is at the end of the sentence --> post description labels.")

            # Assign to the subcaptions the related sentences.
            _post_labels(subcaptions=subcaptions,
                         subcapt=caption_sentence,
                         positions=positions)

        # Check if the first label extracted is at the beginning of the sentence.
        elif positions[0].start_index == 0:
            # Consider labels as 'pre description' labels.

            print("first extracted label is at the beginning of the sentence --> pre description labels.")

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
                subsentence = subsentence + ';'

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


def extract_subcaptions(caption: str,
                        filtered_labels: List[str],
                        target_regex: re.Pattern,
                        target_regex_pos: re.Pattern):
    """
    Split the caption in sentences.

    Args:
        caption (str):                  The caption to split.
        filtered_labels (List[str]):    A list of labels to guide the splitting process.
        target_regex (re.Pattern):      Regex corresponding to identified labels.
        target_regex_POS (re.Pattern):  Regex corresponding to identified POS.
    """
    # Initialize the dictionaries containing the subcaptions.
    subcaptions: Dict[str, str] = {label: '' for label in filtered_labels}
    fuzzy_captions = {label: '' for label in filtered_labels}

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
    image_pointers: List[str] = []

    # Loop through all the sentences of the caption after preface:
    for sentence_index, caption_sentence in enumerate(caption_sentences_list[preface_end_index:]):
        # For each substring extract all the image pointers (labels) and their positions.

        # TODO remove. No actual need for index.
        print("\nsentence_index :", sentence_index)
        print("caption_sentence :", caption_sentence)


        _process_caption_sentence(caption_sentence=caption_sentence,
                                  subcaptions=subcaptions,
                                  fuzzy_captions=fuzzy_captions,
                                  image_pointers=image_pointers,
                                  filtered_labels=filtered_labels,
                                  target_regex=target_regex,
                                  target_regex_pos=target_regex_pos)

    # Assign each element in fuzzycaptions to subcaptions, labelling it as 'fuzzy'.
    for label, value in fuzzy_captions.items():
        subcaptions[label] += ' FUZZY: ' + value

    return subcaptions
