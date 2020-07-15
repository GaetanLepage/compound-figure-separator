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

Collaborator:   Niccolò Marini (niccolo.marini@hevs.ch)


##################################################################
Caption splitting script.

The original version of this code was written by Stefano Marchesin
(stefano.marchesin@unipd.it).

TODO split in multiple files
"""

import sys
import getopt
import re
import json
from typing import List, Dict, Tuple

import nltk

# TODO uncomment
# from .replaceutf8 import replace_utf8
# from . import regex_definitions

# TODO remove (switch back to relative imports)
from compfigsep.utils.figure.labels_structure import LC_ROMAN, UC_ROMAN
from compfigsep.caption_splitting.replaceutf8 import replace_utf8
from compfigsep.caption_splitting import regex_definitions


# A position is a tuple with the following format:
#  (start index,
#   end_index,
#   string)
# TODO define better
Position = Tuple[int, int, str]


# TODO function for script
def get_index(caption_file_lines: List[str],
              target_url: str
              ) -> Tuple[int, str]:
    """
    Associate the ground truth caption with the caption within the csv file.

    Args:
        caption_file_lines (List[str]): TODO.
        target_url (str):               TODO.

    Returns:
        index (int):    TODO.
        caption (str):  TODO.
    """
    for index, caption_file_line in enumerate(caption_file_lines):
        print(caption_file_line)
        url, caption = caption_file_line.strip().split('\t')
        if url == target_url:
            return index, caption

    print("No match")
    print(caption_file_lines)
    print(target_url)


# TODO func for step 4
def sentence_preface(splitted_sentence: str,
                     target_regex: re.Pattern = regex_definitions.RE_CHARACTERS,
                     range_regex1: re.Pattern = regex_definitions.RE_CONJUNCTIONS,
                     range_regex2: re.Pattern = regex_definitions.RE_HYPHEN
                     ) -> Tuple[str, int]:
    """
    Set the preface.
    TODO

    Args:
        splitted_sentence (str):    TODO.
        target_regex (re.Pattern):  TODO.
        range_regex1 (re.Pattern):  TODO.
        range_regex2 (re.Pattern):  TODO.

    Returns:
        preface (str): TODO.
        counter (int): TODO.
    """
    # Define the preface string.
    preface = ''
    # Define preface_pos to set the delimiter for preface.
    preface_pos = []
    counter = 0

    max_counter = len(splitted_sentence)
    # check beginning sentences that do not contain any image pointer
    while not preface_pos and counter < max_counter:
        for pos in range_regex1.finditer(splitted_sentence[counter]):
            preface_pos.append((pos.start(),
                                pos.end(),
                                pos.group(0)))

        for pos in range_regex2.finditer(splitted_sentence[counter]):
            preface_pos.append((pos.start(),
                                pos.end(),
                                pos.group(0)))

        for pos in target_regex.finditer(splitted_sentence[counter]):
            preface_pos.append((pos.start(),
                                pos.end(),
                                pos.group(0)))

        # If preface_pos does not contain anything then append the sentence to preface and update
        # counter.
        if not preface_pos:
            preface += splitted_sentence[counter] + ' '
            counter += 1

    # If the script cannot find a preface, the first sentence is the preface
    if counter == max_counter:
        preface = splitted_sentence[0]

    return preface, counter


# TODO func for step 4
def label_positions(subcaption: str,
                    target_regex: re.Pattern = regex_definitions.RE_CHARACTERS,
                    range_regex1: re.Pattern = regex_definitions.RE_CONJUNCTIONS,
                    range_regex2: re.Pattern = regex_definitions.RE_HYPHEN
                    ) -> List[Position]:
    """
    Set the positions of labels within the sentence
    TODO

    Args:
        subcaption (str):           TODO.
        target_regex (re.Pattern):  TODO.
        range_regex1 (re.Pattern):  TODO.
        range_regex2 (re.Pattern):  TODO.

    Returns:
        positions (List[Position]): TODO.
    """
    # Loop through all the regex (i.e. char, hyphen and conj) and put them into
    # positions.
    positions = []

    # Conjunctions.
    for pos in range_regex1.finditer(subcaption):
        range_expnd = []
        # Expand the range into a list of image pointers.
        range_cleaned = re.sub(pattern=r'[().:,]',
                               repl=' ',
                               string=pos.group(0).replace('and', ' '))

        # Only keep labels containing only alphanumerical characters.
        range_expnd += list(label
                            for label in range_cleaned
                            if label.isalnum())

        positions.append((pos.start(),
                          pos.end(),
                          range_expnd))

    # Hyphen.
    for pos in range_regex2.finditer(subcaption):
        range_expnd = []
        # Expand the range into a list of image pointers.
        range_cleaned = re.sub(pattern=r'[().:]',
                               repl='',
                               string=pos.group(0))
        # Check if the range is numerical or alphabetical.
        if any(d.isdigit() for d in range_cleaned):
            range_expnd += list(map(int,
                                    range(ord(range_cleaned[0]),
                                          ord(range_cleaned[-1]) + 1)
                                    )
                                )
        else:
            range_expnd += list(map(func=chr,
                                    iterables=range(ord(range_cleaned[0]),
                                                    ord(range_cleaned[-1]) + 1)
                                    )
                                )

        positions.append((pos.start(),
                          pos.end(),
                          range_expnd))

    # Target labels.
    for pos in target_regex.finditer(subcaption):
        # clean single labels from additional elements
        char_cleaned = [re.sub(pattern=r'[().:,]',
                               repl='',
                               string=pos.group(0))]

        positions.append((pos.start(),
                          pos.end(),
                          char_cleaned))
    positions.sort()

    return positions


# TODO func for step 4
def POS_positions(subcaption: str,
                  target_regex_POS: re.Pattern = regex_definitions.RE_CHARACTERS_POS,
                  range_regex1_POS: re.Pattern = regex_definitions.RE_CONJUNCTIONS_POS,
                  range_regex2_POS: re.Pattern = regex_definitions.RE_HYPHEN_POS
                  ) -> List[Position]:
    """
    Set the positions of POS labels within the sentence.

    Args:
        subcaption (str):               The subcaption text.
        target_regex_POS (re.Pattern):  TODO.
        range_regex1_POS (re.Pattern):  TODO.
        range_regex2_POS (re.Pattern):  TODO.

    Returns:
        positions_POS (List[Position]): A list of tuple representing TODO.
    """
    # Loop through all the POS regex (i.e. char, hyphen and conj) and put them into positions_POS.
    positions_POS = []
    # Conjunctions
    for pos in range_regex1_POS.finditer(subcaption):
        # There is no need to expand the range as we are only interested in its position.
        positions_POS.append((pos.start(),
                              pos.end(),
                              pos.group(0)))
    # hyphen
    for pos in range_regex2_POS.finditer(subcaption):
        # there is no need to expand the range as we are only interested in its position.
        positions_POS.append((pos.start(),
                              pos.end(),
                              pos.group(0)))
    # target labels
    for pos in target_regex_POS.finditer(subcaption):
        positions_POS.append((pos.start(),
                              pos.end(),
                              pos.group(0)))
    positions_POS.sort()

    return positions_POS


# TODO func for step 4
def remove_overlaps(positions: List[Position]):
    """
    Remove overlapping elements (e.g. (A-D) and D)).

    Args:
        positions (List[Position]): TODO.
    """
    # Remove overlapping elements (e.g. (A-D) and D)).
    for index, pos in enumerate(positions):

        # Skip the last element of the list.
        if index != len(positions) - 1:

            # Check if the end delimiter is greater than or equal to the following one.
            if positions[index][1] >= positions[index + 1][1]:

                # If true then remove the index + 1 element since it is redundant.
                positions.pop(index + 1)


# TODO func for step 4
def remove_POS(positions: List[Position],
               positions_POS):
    """
    Remove from 'positions' elements that have been classified as POS labels.

    Args:
        positions (List[Position]): TODO.
        positions_POS (TODO):       TODO.
    """
    # Check for words that are associated to labels like in, from and panel within the sentence.
    # Check for elements within positions_POS that incorporate positions elements.
    for pos_POS in positions_POS:
        # check if the ending delimiter of an image pointer is equal to that of a POS label.
        for index, pos in enumerate(positions):
            if pos_POS[1] == pos[1]:
                # If the two ending delimiters are equal then remove from positions the image.
                # pointer (POS)
                positions.pop(index)


# TODO func for step 4
def post_labels(subcaptions,
                subcapt,
                positions):
    """
    Associate post description labels to sentences.

    Args:
        subcaptions (TODO):         TODO.
        subcapt (TODO):             TODO.
        positions (List[Position]): TODO.
    """
    # Loop through all the labels detected within the sentence.
    for index, pos in enumerate(positions):
        if index == 0:
            end = positions[index][0]
            # Avoid wrong cases like (A-D) ______ (B)____(C). Where (A-D) is clearly in an
            # incorrect position.
            if end != 0:
                # Loop through the list of labels attached to each position.
                label_list = positions[index][2]
                for label in label_list:
                    # Inserted try catch for avoiding error when the script misleads (i,v) for
                    # roman numbers.
                    try:
                        subcaptions[label] += subcapt[:end] + '. '
                    except:
                        pass
        else:
            # Initial position equal to end delimiter of (index -1) + 1.
            init = positions[index - 1][1]

            # Ending position equal to initial delimiter of index.
            end = positions[index][0]

            # Loop through the list of labels attached to each position.
            label_list = positions[index][2]
            for label in label_list:
                try:
                    subcaptions[label] += subcapt[init:end] + '. '
                except:
                    pass


# TODO func for step 4
def pre_labels(subcaptions,
               subcapt,
               positions: List[Position]):
    """
    Associate pre description labels to sentences
    TODO

    Args:
        subcaptions (TODO): TODO.
        subcapt (TODO):     TODO.
        positions (TODO):   TODO.
    """
    # Loop through all the labels detected within the sentence.
    for index, pos in enumerate(positions):
        if index == len(positions) - 1:
            init = positions[index][1]

            # Avoid wrong cases like (A) ____ (B) ____(C). Where (C) is clearly
            # in an incorrect position.
            if init != sentence_len:
                # loop through the list of labels attached to each position
                label_list = positions[index][2]
                for label in label_list:
                    subcaptions[label] += subcapt[init:] + '. '
        else:
            # initial position equal to end delimiter of index + 1
            init = positions[index][1]
            # ending position equal to initial delimiter of (index + 1)
            end = positions[index + 1][0]
            # loop through the list of labels attached to each position
            label_list = positions[index][2]
            for label in label_list:
                subcaptions[label] += subcapt[init:end] + '. '


# TODO func for step 4
def clean_positions(positions: List[Position],
                    selected_labels):
    """
    TODO
    """
    new_positions = []
    for position in positions:
        list_labels_expanded = position[2]

        if all(label in selected_labels
               for label in list_labels_expanded):

            new_positions.append(position)

    return new_positions


########
# STEPS
########


## 4th step: subcaption extraction
def extract_subcaptions(caption: str,
                        filtered_labels: List[str]):
    """
    Split the caption in sentences.

    Args:
        caption (str):                  The caption to split.
        filtered_labels (List[str]):    A list of labels to guide the splitting process.
    """
    # TODO remove
    print("\n## Step 4")
    print("filtered_labels = ", filtered_labels)

    # Initialize the dictionary containing the subcaptions
    subcaptions = {label: '' for label in filtered_labels}
    fuzzycaptions = {label: '' for label in filtered_labels}

    caption_split = nltk.sent_tokenize(caption)
    # CAVEAT: Check from step 3 which type labels are.
    #         ==> Regular expressions have to be used accordingly (chars, romans or digits)
    # Obtain the preface string and the counter to use for starting to consider
    # non-preface sentences.
    preface_counter = sentence_preface(splitted_sentence=caption_split)

    preface = preface_counter[0]

    # CAVEAT: remember to check whether or not the preface (when it's not empty) matches one of
    # the image pointers extracted previously, otherwise remove false positives from it.

    # If preface is not an empty string then the preface has to be associated to each
    # subpanel caption.
    if preface != '':
        # associate preface to each subcaption
        for key, subcaption in subcaptions.items():
            subcaptions[key] += preface

    # set the starter point for looping through the sentences to counter
    starter = preface_counter[1]
    # define the list of image pointers that each sentence contains
    image_pointers = []


    # Loop through all the sentences of the caption after preface:
    for index_subcapt, subcapt in enumerate(caption_split[starter:]):
    # For each substring extract all the image pointers and their positions.
        # Define the list of tuples representing image pointers.
        positions = []
        # Define the list of tuples representing POS labels
        positions_POS = []
        # loop through all the regex (i.e. char, hyphen and conj) and put them into positions.
        positions = label_positions(subcaption=subcapt)

        # loop through all the POS regex (i.e. char, hyphen and conj) and put them into
        # positions_POS.
        positions_POS = POS_positions(subcaption=subcapt)

        # remove overlapping elements (e.g. (A-D) and D))
        remove_overlaps(positions)

        positions = clean_positions(positions=positions,
                                   selected_labels=filtered_labels)

        # print(positions)
        # compute the length of each sentence
        sentence_len = len(subcapt)
        # check if positions list is empty or not
        if positions:
            # assign to image_pointers the list of labels for the sentence (it will be kept until
            # a new sentence with labels won't be found)
            temp = []
            for pos in positions:
                temp = temp + pos[2]

            image_pointers = list(set(temp))

            # Classify labels in pre, post and in description.
            # Check if positions_POS is empty or not.
            if positions_POS:
                # Check if the last label extracted is at the end of the sentence and it is not
                # contained in positions_POS.
                if positions[-1][1] == sentence_len and positions[-1][1] != positions_POS[-1][1]:
                # Consider labels as post description labels
                    # Check for words that are associated to labels like in, from and panel
                    # within the sentence.
                    remove_POS(positions, positions_POS)
                    # assign to the subcaptions the related sentences
                    post_labels(subcaptions=subcaptions,
                                subcapt=subcapt,
                                positions=positions)

                # Check if the first label extracted is at the beginning of the sentence.
                elif positions[0][0] == 0:
                # Consider labels as pre description labels.

                    # Check for words that are associated to labels like in, from and panel
                    # within the sentence.
                    remove_POS(positions, positions_POS)

                    # Assign to the subcaptions the related sentences.
                    pre_labels(subcaptions, subcapt, positions)

                # Consider labels as in descriptions labels.
                else:
                    # Split the sentence according to ;
                    splitted_sentence = re.split(';', subcapt)

                    # add ; to each element but the last
                    for subsentence in splitted_sentence[:-1]:
                        subsentence += ';'

                    # Obtain all the labels of the sentence from positions.
                    sub_labels = []
                    for element in positions:
                        # Concatenate the different lists contained in positions.
                        sub_labels += element[2]

                    # Remove unnecessary duplicates.
                    sub_labels = list(set(sub_labels))

                    # Sort the labels.
                    sub_labels.sort()

                    # Obtain the preface string and the counter to use for starting to consider
                    # non-preface sentences.
                    sub_preface_counter = sentence_preface(splitted_sentence=splitted_sentence)

                    sub_preface = sub_preface_counter[0]
                    # If preface is not an empty string then the preface has to be associated to
                    # subpanel captions within sub_labels.
                    if sub_preface != '':
                        # Associate preface to each subcaption contained in sub_labels.
                        for key in sub_labels:
                            subcaptions[key] = subcaptions[key] + sub_preface
                    # Set the starter point for looping through the subsentences to counter.
                    sub_starter = sub_preface_counter[1]
                    # Define the list of image pointers that each subsentence contains.
                    sub_image_pointers = []
                    # Loop through all the subsentences of the sentence after preface.
                    for index_subsent, subsent in enumerate(splitted_sentence[sub_starter:]):
                    # For each subsentence extract all the image pointers and their positions.

                        # Define the list of tuples representing image pointers.
                        sub_positions = []

                        # Define the list of tuples representing POS labels.
                        sub_positions_POS = []

                        # Loop through all the regex (i.e. char, hyphen and conj) and put them
                        # into sub_positions.
                        sub_positions = label_positions(subcaption=subsent)

                        # Loop through all the POS regex (i.e. char, hyphen and conj) and put
                        # them into sub_positions_POS.
                        sub_positions_POS = POS_positions(subcaption=subsent)

                        # Remove overlapping elements (e.g. (A-D) and D)).
                        remove_overlaps(sub_positions)
                        # Compute the length of each subsentence.
                        subsentence_len = len(subsent)

                        # Check if sub_positions list is empty or not.
                        if sub_positions:

                            # Assign to sub_image_pointers the list of labels for the subsentence
                            # (it will be kept until a new subsentence with labels won't be
                            # found).
                            temp = []
                            for pos in sub_positions:
                                temp += pos[2]

                            sub_image_pointers = list(set(temp))
                            # Classify labels in pre, post and in description
                            # Check if sub_positions_POS is empty or not.
                            if sub_positions_POS:
                                # Check if the last label extracted is at the end of the
                                # subsentence and it is not contained in sub_positions_POS.
                                if sub_positions[-1][1] == subsentence_len\
                                    and sub_positions[-1][1] != sub_positions_POS[-1][1]:
                                # Consider labels as post description labels.

                                    # Check for words that are associated to labels like in, from
                                    # and panel within the subsentence.
                                    remove_POS(sub_positions, sub_positions_POS)

                                    # Assign to the subcaptions the related subsentences.
                                    post_labels(subcaptions=subcaptions,
                                                subcapt=subsent,
                                                positions=sub_positions)

                                # Check if the first label extracted is at the beginning of the
                                # sentence.
                                elif sub_positions[0][0] == 0:
                                # Consider labels as pre description labels.

                                    # Check for words that are associated to labels like in, from
                                    # and panel within the subsentence.
                                    remove_POS(sub_positions, sub_positions_POS)

                                    # Assign to the subcaptions the related sentences.
                                    pre_labels(subcaptions, subsent, sub_positions)

                                # Consider labels as 'fuzzy labels' and store the subsentence in
                                # fuzzycaptions.
                                else:
                                    for key in sub_labels:
                                        fuzzycaptions[key] = fuzzycaptions[key] + subsent + ' '
                            else:
                                # Check if the last label extracted is at the end of the
                                # subsentence.
                                if sub_positions[-1][1] == subsentence_len:
                                # Consider labels as post description labels.
                                    # Assign to the subcaptions the related sentences.
                                    post_labels(subcaptions, subsent, sub_positions)

                                # Check if the first label extracted is at the beginning of the
                                # subsentence.
                                elif sub_positions[0][0] == 0:
                                # Consider labels as pre description labels.

                                    # Assign to the subcaptions the related subsentences.
                                    pre_labels(subcaptions, subsent, sub_positions)

                                # Consider labels as 'fuzzy labels' and store the subsentence in
                                # fuzzycaptions.
                                else:
                                    for key in sub_labels:
                                        fuzzycaptions[key] = fuzzycaptions[key] + subsent + ' '

                        # Assign the subsentence without labels to the subcaptions that have been
                        # expanded in the previous iteration.
                        else:
                            for label in sub_image_pointers:
                                subcaptions[label] = subcaptions[label] + subsent

            else:
                # Check if the last label extracted is at the end of the sentence.
                if positions[-1][1] == sentence_len:
                # Consider labels as post description labels.

                    # Assign to the subcaptions the related sentences.
                    post_labels(subcaptions=subcaptions,
                                subcapt=subcapt,
                                positions=positions)

                # Check if the first label extracted is at the beginning of the sentence.
                elif positions[0][0] == 0:

                # Consider labels as pre description labels.

                    # Assign to the subcaptions the related sentences.
                    pre_labels(subcaptions, subcapt, positions)

                # Consider labels as in descriptions labels.
                else:
                    # Split the sentence according to ;
                    splitted_sentence = re.split(pattern=';',
                                                 string=subcapt)

                    # Add ; to each element but the last.
                    for subsentence in splitted_sentence[:-1]:
                        subsentence = subsentence + ';'

                    # Obtain all the labels of the sentence from positions.
                    sub_labels = []
                    for element in positions:
                        # concatenate the different lists contained in positions
                        sub_labels = sub_labels + element[2]

                    # Remove unnecessary duplicates.
                    sub_labels = list(set(sub_labels))

                    # Sort the labels.
                    sub_labels.sort()

                    # Obtain the preface string and the counter to use for starting to consider
                    # non-preface sentences.
                    sub_preface_counter = sentence_preface(splitted_sentence=splitted_sentence)

                    sub_preface = sub_preface_counter[0]

                    # If preface is not an empty string then the preface has to be associated to
                    # subpanel captions within sub_labels.
                    if sub_preface != '':
                        # Associate preface to each subcaption contained in sub_labels.
                        for key in sub_labels:
                            subcaptions[key] += sub_preface

                    # Set the starter point for looping through the subsentences to counter.
                    sub_starter = sub_preface_counter[1]

                    # Define the list of image pointers that each subsentence contains.
                    sub_image_pointers = []

                    # Loop through all the subsentences of the sentence after preface.
                    for index_subsent, subsent in enumerate(splitted_sentence[sub_starter:]):
                    # For each subsentence extract all the image pointers and their positions.

                        # Define the list of tuples representing image pointers
                        sub_positions = []

                        # Loop through all the regex (i.e. char, hyphen and conj) and put them
                        # into sub_positions.
                        sub_positions = label_positions(subcaption=subsent)

                        # Remove overlapping elements (e.g. (A-D) and D)).
                        remove_overlaps(positions=sub_positions)

                        # Compute the length of each subsentence.
                        subsentence_len = len(subsent)

                        # Check if sub_positions list is empty or not.
                        if sub_positions:
                            # Assign to sub_image_pointers the list of labels for the subsentence.
                            # TODO rephrase:
                            # (It will be kept until a new subsentence with labels won't be found).
                            temp = []
                            for pos in sub_positions:
                                temp += pos[2]

                            sub_image_pointers = list(set(temp))

                            # Classify labels in 'pre', 'post' and 'in' description (remember that
                            # there are no POS in this case).
                            # Check if the last label extracted is at the end of the subsentence.
                            if sub_positions[-1][1] == subsentence_len:
                            # Consider labels as post description labels.

                                # Assign to the subcaptions the related sentences
                                post_labels(subcaptions=subcaptions,
                                            subcapt=subsent,
                                            positions=sub_positions)

                            # Check if the first label extracted is at the beginning of the
                            # subsentence.
                            elif sub_positions[0][0] == 0:
                            # Consider labels as pre description labels.

                                # Assign to the subcaptions the related subsentences.
                                pre_labels(subcaptions=subcaptions,
                                           subcapt=subsent,
                                           positions=sub_positions)

                            # Consider labels as 'fuzzy labels' and store the subsentence in
                            # fuzzycaptions.
                            else:
                                for key in sub_labels:
                                    try:
                                        fuzzycaptions[key] += subsent + ' '
                                    except:
                                        pass

                        # Assign the subsentence without labels to the subcaptions that have been
                        # expanded in the previous iteration.
                        else:
                            for label in sub_image_pointers:
                                try:
                                    subcaptions[label] += subsent
                                except:
                                    pass

        # Assign the sentence without labels to the subcaptions that have been expanded in the
        # previous iteration.
        else:
            for label in image_pointers:
                subcaptions[label] += subcapt
    # assign each element in fuzzycaptions to subcaptions, labelling it as 'fuzzy'
    for label, value in fuzzycaptions.items():
        subcaptions[label] += ' FUZZY: ' + value
        #subcaptions[label] = value + ', ' + subcaptions[label]
    #print("subcaptions")
    #print (subcaptions)

    return subcaptions




def main():
    # TODO parse args


    # argv = sys.argv[1:]


    ## there are six different types of classes that can be used as caption labels
    ## class1 - (a), (A), (1), (I)
    ## class2 - a), A), 1), I)
    ## class3 - a., A., 1., I.
    ## class4 - a:, A:, 1:, I:
    ## class5 - a, A, 1, I
    ## class6 - a1, A1
    # There are also abbrvs. like (a-e) or (a,b and c) that have first to be
    # expanded and then detected.

    ### TODO
    ## preparing functions for steps 1-4
    ## 1st step: label identification
    ## 2nd step: label expansion
    ## 3rd step: label filtering
    ## 4th step: subcaption extraction

    # This parameter is linked to the filename_captions csv file, for selecting
    # which caption to use.
    # try:
        # opts, args = getopt.getopt(argv, "hn:", ["n_exp="])

    # except getopt.GetoptError:
        # print('Student_Model_finetuning_var_pretrained.py -n <n_exp>')
        # sys.exit(2)

    # for opt, arg in opts:
        # if opt == '-h':
            # print('Student_Model_finetuning_var_pretrained.py -n <n_exp>')

            # sys.exit()

        # elif opt in ("-n", "-n_exp"):
            # N_EXP_str = arg

    # TODO: modify the paths
    filename_ground_truth = '/home/gaetan/compfigsep/compfigsep/caption_splitting/csv_folder/captions_ground_truth'
    filename_captions = '/home/gaetan/compfigsep/compfigsep/caption_splitting/csv_folder/captions_prostate.csv'


    with open(filename_ground_truth) as ground_truth:
        ground_truth_data = json.load(ground_truth)

    with open(filename_captions, 'r') as dmli:
        captions = dmli.readlines()

    # image_data['url'] = path to the file in fast
    # image_data['labels'] = labels within the caption
    # image_data['subcaptions'] = subcaptions

    N_EXP_str = "0"
    test_idx = int(N_EXP_str)
    idx, caption = get_index(captions, ground_truth_data[test_idx]['url'])
    selected_labels = ground_truth_data[test_idx]['labels']

    caption = str(replace_utf8(caption))

    print("Caption:", caption)
    print("Selected labels:", selected_labels)

    # Step 1
    # labels = label_identification(caption=caption)

    # Step 2
    # TODO = label_expansion(TODO)

    # Step 3
    # filtered_labels = label_filtering(labels)
    filtered_labels = ["a", "b"]

    # Step 4
    subcaptions = extract_subcaptions(caption=caption,
                                      filtered_labels=filtered_labels)
    #print("ground_truth")
    #print(ground_truth_data[test_idx])

    for label_index, label in enumerate(filtered_labels):
        print("label: " + label)
        print("subcaption:", subcaptions[label])
        print("ground_truth " + label)
        # print(ground_truth_data[test_idx]['subcaptions'][label_index])

if __name__ == '__main__':
    main()
