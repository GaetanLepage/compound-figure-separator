# -*- coding: utf-8 -*-
import sys
import getopt
import re
import json
import nltk

# TODO uncomment
# from ..utils.figure.labels_structure import LC_ROMAN, UC_ROMAN
# from .replaceutf8 import replace_utf8

# TODO remove
from compfigsep.utils.figure.labels_structure import LC_ROMAN, UC_ROMAN
from compfigsep.caption_splitting.replaceutf8 import replace_utf8



argv = sys.argv[1:]


## there are six different types of classes that can be used as caption labels
## class1 - (a), (A), (1), (I)
## class2 - a), A), 1), I)
## class3 - a., A., 1., I.
## class4 - a:, A:, 1:, I:
## class5 - a, A, 1, I
## class6 - a1, A1
## there are also abbrvs. like (a-e) or (a,b and c) that have first to be expanded and then detected

### TODO
## preparing functions for steps 1-4
## 1st step: label identification
## 2nd step: label expansion
## 3rd step: label filtering
## 4th step: subcaption extraction

# this parameter is linked to the filename_captions csv file, for selecting which caption to use
try:
    opts, args = getopt.getopt(argv, "hn:", ["n_exp="])

except getopt.GetoptError:
    print('Student_Model_finetuning_var_pretrained.py -n <n_exp>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('Student_Model_finetuning_var_pretrained.py -n <n_exp>')

        sys.exit()

    elif opt in ("-n", "-n_exp"):
        N_EXP_str = arg

# TODO: modify the paths
filename_ground_truth = '/home/gaetan/hevs/implementations/Caption_Splitting/csv_folder/captions_ground_truth'
filename_captions = '/home/gaetan/hevs/implementations/Caption_Splitting/csv_folder/captions_prostate.csv'


with open(filename_ground_truth) as ground_truth:
    ground_truth_data = json.load(ground_truth)

with open(filename_captions, 'r') as dmli:
    captions = dmli.readlines()


def get_index(caption_file_lines,
              target_url):
    """
    Associate the ground truth caption with the caption within the csv file.

    Args:
        TODO

    Returns:
        TODO
    """
    for index, caption_file_line in enumerate(caption_file_lines):
        print(caption_file_line)
        url, caption = caption_file_line.strip().split('\t')
        if url == target_url:
            return index, caption



# ###### image_data['url'] = path to the file in fast
# ###### image_data['labels'] = labels within the caption
# ###### image_data['subcaptions'] = subcaptions


test_idx = int(N_EXP_str)
idx, caption = get_index(ground_truth_data[test_idx]['url'], captions)
selected_labels = ground_truth_data[test_idx]['labels']

caption3 = str(replace_utf8(caption))

print("CAPTION NUMBER " + N_EXP_str)
print()
print(caption3)
print(selected_labels)

# ##### declare functions, lookup tables and regular expressions for steps 1-4


# define a function that given a latin number converts it to its numerical equivalent and
# viceversa (both for upper/lower cases).


# Build the regular expressions that match classes.
characters = re.compile(r"((\b|\.|\()([a-z]|[A-Z])(?![^).:0-9])[0-9]?(\)|\.|:|\b)?)")
latins = re.compile(r"((\b|\.|\()([i,ii,iii,iv,v,vi,vii,viii,ix,x,xi,xii,xiii,xiv,xv,xvi,xvii,xviii,xix,xx]|[I,II,III,IV,V,VI,VII,VIII,IX,X,XI,XII,XIII,XIV,XV,XVI,XVII,XVIII,XIX,XX])(?![^).:])(\)|\.|:|\b)?)")
digits = re.compile(r"((\b|\.|\()([1-9])(?![^).:0-9])[0-9]?(\)|\.|:|\b)?)")

# Build the regular expressions that match ranges (hyphens and conjunctions).
hyphen = re.compile(r"((\b|\.|\()([a-z]|[A-Z]|[i,ii,iii,iv,v,vi,vii,viii,ix,x,xi,xii,xiii,xiv,xv,xvi,xvii,xviii,xix,xx]|[I,II,III,IV,V,VI,VII,VIII,IX,X,XI,XII,XIII,XIV,XV,XVI,XVII,XVIII,XIX,XX]|[1-9])[0-9]?-([[a-z]|[A-Z]|[i,ii,iii,iv,v,vi,vii,viii,ix,x,xi,xii,xiii,xiv,xv,xvi,xvii,xviii,xix,xx]|[I,II,III,IV,V,VI,VII,VIII,IX,X,XI,XII,XIII,XIV,XV,XVI,XVII,XVIII,XIX,XX]|[1-9])[0-9]?\)?)")
conjunctions = re.compile(r"((\b|\.|\()(([a-z]|[A-Z]|[i,ii,iii,iv,v,vi,vii,viii,ix,x,xi,xii,xiii,xiv,xv,xvi,xvii,xviii,xix,xx]|[I,II,III,IV,V,VI,VII,VIII,IX,X,XI,XII,XIII,XIV,XV,XVI,XVII,XVIII,XIX,XX]|[1-9])[0-9]?(\)|\.|:)?)((\s?,\s?\(?([a-z]|[A-Z]|[i,ii,iii,iv,v,vi,vii,viii,ix,x,xi,xii,xiii,xiv,xv,xvi,xvii,xviii,xix,xx]|[I,II,III,IV,V,VI,VII,VIII,IX,X,XI,XII,XIII,XIV,XV,XVI,XVII,XVIII,XIX,XX]|[1-9])[0-9]?(\)|\.|:)?)+|(\s?and\s?\(?([a-z]|[A-Z]|[i,ii,iii,iv,v,vi,vii,viii,ix,x,xi,xii,xiii,xiv,xv,xvi,xvii,xviii,xix,xx]|[I,II,III,IV,V,VI,VII,VIII,IX,X,XI,XII,XIII,XIV,XV,XVI,XVII,XVIII,XIX,XX]|[1-9])[0-9]?(\)|\.|:)?\W)+)(\s?,?\s?and\s?\(?([a-z]|[A-Z]|[i,ii,iii,iv,v,vi,vii,viii,ix,x,xi,xii,xiii,xiv,xv,xvi,xvii,xviii,xix,xx]|[I,II,III,IV,V,VI,VII,VIII,IX,X,XI,XII,XIII,XIV,XV,XVI,XVII,XVIII,XIX,XX]|[1-9])[0-9]?(\)|\.|:)?\W)?\W\)?)")

# Define the regular expressions that match Part Of Speech (POS) labels.
# => single labels
chars_POS = re.compile(r"((in|from|panel(s)?)\s?\(?([a-z]|[A-Z])(?![^).:0-9])[0-9]?(\)|\.|:|\b)?)")
latins_POS = re.compile(r"((in|from|panel(s)?)\s?\(?([i,ii,iii,iv,v,vi,vii,viii,ix,x,xi,xii,xiii,xiv,xv,xvi,xvii,xviii,xix,xx]|                         [I,II,III,IV,V,VI,VII,VIII,IX,X,XI,XII,XIII,XIV,XV,XVI,XVII,XVIII,XIX,XX])(?![^).:])(\)|\.|:|\b)?)")
digits_POS = re.compile(r"((in|from|panel(s)?)\s?\(?([1-9])(?![^).:0-9])[0-9]?(\)|\.|:|\b)?)")
# => range labels
hyphen_POS = re.compile(r"((in|from|panel(s)?)\s?\(?([a-z]|[A-Z]|[i,ii,iii,iv,v,vi,vii,viii,ix,x,xi,xii,xiii,xiv,xv,xvi,xvii,xviii,xix,xx]|[I,II,III,IV,V,VI,VII,VIII,IX,X,XI,XII,XIII,XIV,XV,XVI,XVII,XVIII,XIX,XX]|[1-9])[0-9]?-([[a-z]|[A-Z]|[i,ii,iii,iv,v,vi,vii,viii,ix,x,xi,xii,xiii,xiv,xv,xvi,xvii,xviii,xix,xx]|[I,II,III,IV,V,VI,VII,VIII,IX,X,XI,XII,XIII,XIV,XV,XVI,XVII,XVIII,XIX,XX]|[1-9])[0-9]?\)?)")

conj_POS = re.compile(r"((in|from|panel(s)?)\s?\(?(([a-z]|[A-Z]|[i,ii,iii,iv,v,vi,vii,viii,ix,x,xi,xii,xiii,xiv,xv,xvi,xvii,xviii,xix,xx]|[I,II,III,IV,V,VI,VII,VIII,IX,X,XI,XII,XIII,XIV,XV,XVI,XVII,XVIII,XIX,XX]|[1-9])[0-9]?(\)|\.|:)?)((\s?,\s?\(?([a-z]|[A-Z]|[i,ii,iii,iv,v,vi,vii,viii,ix,x,xi,xii,xiii,xiv,xv,xvi,xvii,xviii,xix,xx]|[I,II,III,IV,V,VI,VII,VIII,IX,X,XI,XII,XIII,XIV,XV,XVI,XVII,XVIII,XIX,XX]|[1-9])[0-9]?(\)|\.|:)?)+|(\s?and\s?\(?([a-z]|[A-Z]|[i,ii,iii,iv,v,vi,vii,viii,ix,x,xi,xii,xiii,xiv,xv,xvi,xvii,xviii,xix,xx]|[I,II,III,IV,V,VI,VII,VIII,IX,X,XI,XII,XIII,XIV,XV,XVI,XVII,XVIII,XIX,XX]|[1-9])[0-9]?(\)|\.|:)?\W)+)(\s?,?\s?and\s?\(?([a-z]|[A-Z]|[i,ii,iii,iv,v,vi,vii,viii,ix,x,xi,xii,xiii,xiv,xv,xvi,xvii,xviii,xix,xx]|[I,II,III,IV,V,VI,VII,VIII,IX,X,XI,XII,XIII,XIV,XV,XVI,XVII,XVIII,XIX,XX]|[1-9])[0-9]?(\)|\.|:)?\W)?\W\)?)")


def is_roman(string: str) -> bool:
    """
    Check if the given string is a roman number (up to 20).
    (i, ii,..., xx).

    Args:
        string (str):   A string that may be a latin number.

    Returns:
        is_roman (bool):    Whether the given string is a latin number.
    """
    return string in UC_ROMAN or string in LC_ROMAN


def sentence_preface(splitted_sentence,
                     target_regex,
                     range_regex1,
                     range_regex2):
    """
    Set the preface.
    TODO

    Args:
        splitted_sentence (TODO):   TODO
        target_regex (TODO):        TODO
        range_regex1 (TODO):        TODO
        range_regex2 (TODO):        TODO

    Returns:
        preface (TODO): TODO
        counter (TODO): TODO
    """
    # define the preface string
    preface = ''
    # define preface_pos to set the delimiter for preface
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
            preface = preface + splitted_sentence[counter] + ' '
            counter = counter + 1

    # If the script cannot find a preface, the first sentence is the preface
    if counter == max_counter:
        preface = splitted_sentence[0]

    return preface, counter


def label_positions(subcaption,
                    target_regex,
                    range_regex1,
                    range_regex2):
    """
    Set the positions of labels within the sentence
    TODO

    Args:
        TODO

    Returns:
        TODO
    """
    # loop through all the regex (i.e. char, hyphen and conj) and put them into positions
    positions = []
    # conjunctions
    for pos in range_regex1.finditer(subcaption):
        range_expnd = []
        # expand the range into a list of image pointers
        range_cleaned = re.sub(r'[().:,]', ' ', pos.group(0).replace('and', ' '))
        range_expnd = range_expnd + list(l for l in range_cleaned if l.isalnum())
        positions.append((pos.start(), pos.end(), range_expnd))

    # hyphen
    for pos in range_regex2.finditer(subcaption):
        range_expnd = []
        # expand the range into a list of image pointers
        range_cleaned = re.sub(r'[().:]', '', pos.group(0))
        # check if the range is numerical or alphabetical
        if any(d.isdigit() for d in range_cleaned):
            range_expnd = range_expnd + list(map(int,
                                                 range(ord(range_cleaned[0]),
                                                       ord(range_cleaned[-1]) + 1)
                                                 )
                                             )
        else:
            range_expnd = range_expnd + list(map(func=chr,
                                                 iterables=range(ord(range_cleaned[0]),
                                                                 ord(range_cleaned[-1]) + 1)
                                                 )
                                             )

        positions.append((pos.start(),
                          pos.end(),
                          range_expnd))

    # target labels
    for pos in target_regex.finditer(subcaption):
        # clean single labels from additional elements
        char_cleaned = [re.sub(r'[().:,]', '', pos.group(0))]
        positions.append((pos.start(),
                          pos.end(),
                          char_cleaned))
    positions.sort()
    return positions


def POS_positions(subcaption,
                  target_regex_POS,
                  range_regex1_POS,
                  range_regex2_POS):
    """
    Set the positions of POS labels within the sentence.

    Args:
        subcaption (TODO):          TODO
        target_regex_POS (TODO):    TODO
        range_regex1_POS (TODO):    TODO
        range_regex2_POS (TODO):    TODO

    Returns:
        TODO
    """
    # Loop through all the POS regex (i.e. char, hyphen and conj) and put them into positions_POS
    positions_POS = []
    # Conjunctions
    for pos in range_regex1_POS.finditer(subcaption):
        # There is no need to expand the range as we are only interested in its position.
        positions_POS.append((pos.start(),
                              pos.end(),
                              pos.group(0)))
    # hyphen
    for pos in range_regex2_POS.finditer(subcaption):
        # there is no need to expand the range as we are only interested in its position
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


def remove_overlaps(positions):
    """
    Remove overlapping elements (e.g. (A-D) and D)).

    Args:
        TODO

    Returns:
        TODO
    """
    # remove overlapping elements (e.g. (A-D) and D))
    for index, pos in enumerate(positions):
        # skip the last element of the list
        if index != len(positions) - 1:
            # check if the end delimiter is greater than or equal to the following one
            if positions[index][1] >= positions[index + 1][1]:
                # if true then remove the index + 1 element since it is redundant
                positions.pop(index + 1)

    return positions


def remove_POS(positions,
               positions_POS):
    """
    Remove from 'positions' elements that have been classified as POS labels.

    Args:
        TODO

    Returns:
        TODO
    """
    # Check for words that are associated to labels like in, from and panel within the sentence.
    # Check for elements within positions_POS that incorporate positions elements.
    for pos_POS in positions_POS:
        # check if the ending delimiter of an image pointer is equal to that of a POS label
        for index, pos in enumerate(positions):
            if pos_POS[1] == pos[1]:
                # If the two ending delimiters are equal then remove from positions the image
                # pointer (POS)
                positions.pop(index)

    return positions


def post_labels(subcaptions, subcapt, positions):
    """
    Associate post description labels to sentences.

    Args:
        TODO

    Returns:
        TODO
    """
    # loop through all the labels detected within the sentence
    for index, pos in enumerate(positions):
        if index == 0:
            end = positions[index][0]
            # avoid wrong cases like (A-D) ______ (B)____(C). Where (A-D) is clearly in an incorrect position
            if end != 0:
                # loop through the list of labels attached to each position
                label_list = positions[index][2]
                for label in label_list:
                    #inserted try catch for avoiding error when the script misleads (i,v) for latin chars
                    try:
                        subcaptions[label] = subcaptions[label] + subcapt[:end] + '. '
                    except:
                        pass
        else:
            # initial position equal to end delimiter of (index -1) + 1
            init = positions[index - 1][1]
            # ending position equal to initial delimiter of index
            end = positions[index][0]
            # loop through the list of labels attached to each position
            label_list = positions[index][2]
            for label in label_list:
                try:
                    subcaptions[label] = subcaptions[label] + subcapt[init:end] + '. '
                except:
                    pass
    return subcaptions


def pre_labels(subcaptions,
               subcapt,
               positions):
    """
    Associate pre description labels to sentences
    TODO

    Args:
        subcaptions (TODO): TODO
        subcapt (TODO):     TODO
        positions (TODO):   TODO

    Returns:
        subcaptions (TODO): TODO
    """
    # Loop through all the labels detected within the sentence.
    for index, pos in enumerate(positions):
        if index == len(positions) - 1:
            init = positions[index][1]
            # Avoid wrong cases like (A) ____ (B) ____(C). Where (C) is clearly in an incorrect
            # position.
            if init != sentence_len:
                # loop through the list of labels attached to each position
                label_list = positions[index][2]
                for label in label_list:
                    subcaptions[label] = subcaptions[label] + subcapt[init:] + '. '
        else:
            # initial position equal to end delimiter of index + 1
            init = positions[index][1]
            # ending position equal to initial delimiter of (index + 1)
            end = positions[index + 1][0]
            # loop through the list of labels attached to each position
            label_list = positions[index][2]
            for label in label_list:
                subcaptions[label] = subcaptions[label] + subcapt[init:end] + '. '

    return subcaptions


def clean_position(positions,
                   selected_labels):
    """
    TODO
    """
    new_positions = []
    for position in positions:
        list_labels_expanded = position[2]
        b = True

        for label in list_labels_expanded:

            if label not in selected_labels:
                b = False

        if b:
            new_positions.append(position)
    return new_positions


# REMOVED THIS PART: instead of searching labels within the string ground truth/detected labels are used

"""
## 1st step: label identification
# detect alphanumerical labels
characters_raw = characters.findall(caption3) #caption3 is the current caption that is being analyzed
characters_cleaned = []
if characters_raw:
    # get the list of alphanumerical labels
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
    print("characters_cleaned")
    print (characters_cleaned)

# detect latin labels
latins_raw = latins.findall(caption3)
latins_cleaned = []
if latins_raw:
    # get the list of latin labels
    latins_list = []
    for raw in latins_raw:
        latins_list.append(raw[0])
    # clean the list
    for element in latins_list:
        latins_cleaned.append(re.sub(r'[().:]', '', element))
    # remove duplicates
    latins_cleaned = list(set(latins_cleaned))
    # check if latin numbers are lower or upper case
    is_upper = latins_cleaned[0].isupper() #thanks to how regular expressions operate on '|' we just need to check first element
    # convert latin to numerical
    for key, value in enumerate(latins_cleaned):
        # check if latin numbers are upper or lower case
        if is_upper:
            latins_cleaned[key] = uc_latin_mapper[value]
        else:
            latins_cleaned[key] = lc_latin_mapper[value]
    # sort the list of numerical numbers and revert it back to its latin form
    latins_cleaned.sort()
    for key, value in enumerate(latins_cleaned):
        # check if latin numbers were upper or lower case
        if is_upper:
            latins_cleaned[key] = uc_latin_mapper[value]
        else:
            latins_cleaned[key] = lc_latin_mapper[value]
    print("latins_cleaned")
    print (latins_cleaned)

# detect numerical labels
digits_raw = digits.findall(caption3)
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
    print("digits_cleaned")
    print (digits_cleaned)


#get hyphens and conjunctions
hyphen_range = hyphen.findall(caption3)
conj_range = conjunctions.findall(caption3)

print("hyphen_range")
print(hyphen_range)
print("conj_range")
print(conj_range)


## 2nd step: label expansion
# extract first element of each tuple and replace the tuple with it
ranges = []
# hyphen range
hyphen_vector = []
for hyphen_tuple in hyphen_range:
    hyphen_vector.append(hyphen_tuple[0])
# conjunction range
conj_vector = []
for conj_tuple in conj_range:
    conj_vector.append(conj_tuple[0])
# clean the elements and expand the sequences
# hyphen range
hyphen_cleaned = []
for element in hyphen_vector:
    hyphen_cleaned.append(re.sub(r'[().:]', '', element))
for element in hyphen_cleaned:
    #split the string by hyphen
    element = element.split('-')
    # check if the range is numerical, latin or alphabetical - CAVEAT: set also the latin one
    if all([d.isdigit() for d in element]): #numerical
        ranges = ranges + list(map(int, range(ord(element[0]), ord(element[-1]) + 1)))
    elif all([islatin(r) for r in element]): #latin
        #convert latin numbers into numericals, expand and then re-convert
        for key, value in enumerate(element):
            # check if latin numbers are upper or lower case
            if is_upper:
                element[key] = uc_latin_mapper[value]
            else:
                element[key] = lc_latin_mapper[value]
        # expand the range of numerical numbers and revert it back to its latin form
        latin_range = list(map(int, range(ord(element[0]), ord(element[-1]) + 1)))
        for key, value in enumerate(latin_range):
            # check if latin numbers were upper or lower case
            if is_upper:
                latin_range[key] = uc_latin_mapper[value]
            else:
                latin_range[key] = lc_latin_mapper[value]
        #concatenate the range of latin numbers to the list of ranges
        ranges = ranges + latin_range
    else: #alphabetical
        ranges = ranges + list(map(chr, range(ord(element[0]), ord(element[-1]) + 1)))
# conjunction range
conj_cleaned = []
#clean the identified patterns from useless characters
for element in conj_vector:
    conj_cleaned.append(re.sub(r'[().:,]', ' ', element.replace('and', ' ')))
#append elements to ranges
for element in conj_cleaned:
    ranges = ranges + element.split()
#remove duplicates
ranges = list(set(ranges))

ranges.sort()
print("ranges")
print (ranges)


# In[23]:


# merge the lists containing the expanded ranges and the single labels (union operation between sets)
labels = list(set(ranges) | set(digits_cleaned) | set(latins_cleaned) | set(characters_cleaned))
labels.sort()
# split the labels list into three sublists: digits, alphanumeric & latin
labels_digits = []
labels_latins = []
labels_alphanum = []
for label in labels:
    if label.isdigit(): #store digit
        labels_digits.append(label)
    elif islatin(label): #store latin
        labels_latins.append(label)
    else: #store alphanumerical
        labels_alphanum.append(label)

print("labels_digits")
print (labels_digits)
print("labels_latins")
print (labels_latins)
print("labels_alphanum")
print (labels_alphanum)


## 3rd step: label filtering
# decide which list to consider (digits/latin or alphanumeric) depending on the amount of matched characters between image and caption

### CAVEAT: This part will be done once we have the result list of characters from images ###


labels_final = [] # CAVEAT: this list is the one that will be used once we know which are the correct labels

if (selected_labels==['none']):
    selected_labels = labels_alphanum
"""
# initialize the dictionary containing the subcaptions
subcaptions = {key: '' for key in selected_labels}
fuzzycaptions = {key: '' for key in selected_labels}
print(subcaptions)
print(fuzzycaptions)

if selected_labels == ['none']:
    print(caption3)

else:
    ## 4th step: subcaption extraction
    # split the caption in sentences
    caption_split = nltk.sent_tokenize(caption3)
    #CAVEAT: check from step 3 which type labels are - regular expressions have to be used accordingly (chars, latins or digits)
    # obtain the preface string and the counter to use for starting to consider non-preface sentences
    preface_counter = sentence_preface(caption_split, characters, conjunctions, hyphen)

    preface = preface_counter[0]

    ## CAVEAT: remember to check whether or not the preface (when it's not empty) matches one of the image pointers \
    ## extracted previously, otherwise remove false positives from it

    # if preface is not an empty string then the preface has to be associated to each subpanel caption
    if preface != '':
        # associate preface to each subcaption
        for key, subcaption in subcaptions.items():
            subcaptions[key] = subcaptions[key] + preface

    # set the starter point for looping through the sentences to counter
    starter = preface_counter[1]
    # define the list of image pointers that each sentence contains
    image_pointers = []


    # loop through all the sentences of the caption after preface
    for index_subcapt, subcapt in enumerate(caption_split[starter:]):
    # for each substring extract all the image pointers and their positions
        # define the list of tuples representing image pointers
        positions = []
        # define the list of tuples representing POS labels
        positions_POS = []
        # loop through all the regex (i.e. char, hyphen and conj) and put them into positions.
        positions = label_positions(subcapt, characters, conjunctions, hyphen)
        # loop through all the POS regex (i.e. char, hyphen and conj) and put them into
        # positions_POS.
        positions_POS = POS_positions(subcapt, chars_POS, conj_POS, hyphen_POS)

        # remove overlapping elements (e.g. (A-D) and D))
        positions = remove_overlaps(positions)

        positions = clean_position(positions, selected_labels)

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
                    positions = remove_POS(positions, positions_POS)
                    # assign to the subcaptions the related sentences
                    subcaptions = post_labels(subcaptions, subcapt, positions)
                # check if the first label extracted is at the beginning of the sentence
                elif positions[0][0] == 0:
                # consider labels as pre description labels
                    # Check for words that are associated to labels like in, from and panel
                    # within the sentence.
                    positions = remove_POS(positions, positions_POS)
                    # assign to the subcaptions the related sentences
                    subcaptions = pre_labels(subcaptions, subcapt, positions)
                # consider labels as in descriptions labels
                else:
                    # split the sentence according to ;
                    sentence_splitted = re.split(';', subcapt)
                    # add ; to each element but the last
                    for subsentence in sentence_splitted[:-1]:
                        subsentence = subsentence + ';'
                    # obtain all the labels of the sentence from positions
                    sub_labels = []
                    for element in positions:
                        # Concatenate the different lists contained in positions.
                        sub_labels = sub_labels + element[2]
                    # Remove unnecessary duplicates.
                    sub_labels = list(set(sub_labels))
                    # Sort the labels.
                    sub_labels.sort()
                    # Obtain the preface string and the counter to use for starting to consider
                    # non-preface sentences.
                    sub_preface_counter = sentence_preface(sentence_splitted,
                                                           characters,
                                                           conjunctions,
                                                           hyphen)
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
                    for index_subsent, subsent in enumerate(sentence_splitted[sub_starter:]):
                    # for each subsentence extract all the image pointers and their positions
                        # define the list of tuples representing image pointers
                        sub_positions = []
                        # define the list of tuples representing POS labels
                        sub_positions_POS = []
                        # Loop through all the regex (i.e. char, hyphen and conj) and put them
                        # into sub_positions.
                        sub_positions = label_positions(subsent,
                                                        characters,
                                                        conjunctions,
                                                        hyphen)
                        # Loop through all the POS regex (i.e. char, hyphen and conj) and put
                        # them into sub_positions_POS.
                        sub_positions_POS = POS_positions(subsent,
                                                          chars_POS,
                                                          conj_POS,
                                                          hyphen_POS)
                        # Remove overlapping elements (e.g. (A-D) and D)).
                        sub_positions = remove_overlaps(sub_positions)
                        # Compute the length of each subsentence.
                        subsentence_len = len(subsent)
                        # Check if sub_positions list is empty or not.
                        if sub_positions:
                            # Assign to sub_image_pointers the list of labels for the subsentence
                            # (it will be kept until a new subsentence with labels won't be
                            # found).
                            temp = []
                            for pos in sub_positions:
                                temp = temp + pos[2]
                            sub_image_pointers = list(set(temp))
                            # Classify labels in pre, post and in description
                            # Check if sub_positions_POS is empty or not.
                            if sub_positions_POS:
                                # Check if the last label extracted is at the end of the
                                # subsentence and it is not contained in sub_positions_POS.
                                if sub_positions[-1][1] == subsentence_len\
                                    and sub_positions[-1][1] != sub_positions_POS[-1][1]:
                                # consider labels as post description labels
                                    # check for words that are associated to labels like in, from and panel within the subsentence
                                    sub_positions = remove_POS(sub_positions, sub_positions_POS)
                                    # assign to the subcaptions the related subsentences
                                    subcaptions = post_labels(subcaptions, subsent, sub_positions)
                                # check if the first label extracted is at the beginning of the sentence
                                elif sub_positions[0][0] == 0:
                                # consider labels as pre description labels
                                    # check for words that are associated to labels like in, from and panel within the subsentence
                                    sub_positions = remove_POS(sub_positions, sub_positions_POS)
                                    # assign to the subcaptions the related sentences
                                    subcaptions = pre_labels(subcaptions, subsent, sub_positions)
                                # consider labels as 'fuzzy labels' and store the subsentence in fuzzycaptions
                                else:
                                    for key in sub_labels:
                                        fuzzycaptions[key] = fuzzycaptions[key] + subsent + ' '
                            else:
                                # check if the last label extracted is at the end of the subsentence
                                if sub_positions[-1][1] == subsentence_len:
                                # consider labels as post description labels
                                    # assign to the subcaptions the related sentences
                                    subcaptions = post_labels(subcaptions, subsent, sub_positions)
                                # check if the first label extracted is at the beginning of the subsentence
                                elif sub_positions[0][0] == 0:
                                # consider labels as pre description labels
                                    # assign to the subcaptions the related subsentences
                                    subcaptions = pre_labels(subcaptions, subsent, sub_positions)
                                # consider labels as 'fuzzy labels' and store the subsentence in fuzzycaptions
                                else:
                                    for key in sub_labels:
                                        fuzzycaptions[key] = fuzzycaptions[key] + subsent + ' '
                        # assign the subsentence without labels to the subcaptions that have been expanded in the previous iteration
                        else:
                            for label in sub_image_pointers:
                                subcaptions[label] = subcaptions[label] + subsent

            else:
                # check if the last label extracted is at the end of the sentence
                if positions[-1][1] == sentence_len:
                # consider labels as post description labels
                    # assign to the subcaptions the related sentences
                    subcaptions = post_labels(subcaptions, subcapt, positions)
                # check if the first label extracted is at the beginning of the sentence
                elif positions[0][0] == 0:
                # consider labels as pre description labels
                    # assign to the subcaptions the related sentences
                    subcaptions = pre_labels(subcaptions, subcapt, positions)
                # consider labels as in descriptions labels
                else:
                    # split the sentence according to ;
                    sentence_splitted = re.split(';', subcapt)
                    # add ; to each element but the last
                    for subsentence in sentence_splitted[:-1]:
                        subsentence = subsentence + ';'
                    # obtain all the labels of the sentence from positions
                    sub_labels = []
                    for element in positions:
                        # concatenate the different lists contained in positions
                        sub_labels = sub_labels + element[2]
                    # remove unnecessary duplicates
                    sub_labels = list(set(sub_labels))
                    # sort the label
                    sub_labels.sort()
                    # Obtain the preface string and the counter to use for starting to consider
                    # non-preface sentences.
                    sub_preface_counter = sentence_preface(sentence_splitted,
                                                           characters,
                                                           conjunctions,
                                                           hyphen)
                    sub_preface = sub_preface_counter[0]
                    # If preface is not an empty string then the preface has to be associated to
                    # subpanel captions within sub_labels
                    if sub_preface != '':
                        # associate preface to each subcaption contained in sub_labels
                        for key in sub_labels:
                            subcaptions[key] = subcaptions[key] + sub_preface
                    # set the starter point for looping through the subsentences to counter
                    sub_starter = sub_preface_counter[1]
                    # define the list of image pointers that each subsentence contains
                    sub_image_pointers = []
                    # loop through all the subsentences of the sentence after preface
                    for index_subsent, subsent in enumerate(sentence_splitted[sub_starter:]):
                    # for each subsentence extract all the image pointers and their positions
                        # define the list of tuples representing image pointers
                        sub_positions = []
                        # Loop through all the regex (i.e. char, hyphen and conj) and put them
                        # into sub_positions.
                        sub_positions = label_positions(subsent, characters, conjunctions, hyphen)
                        # remove overlapping elements (e.g. (A-D) and D))
                        sub_positions = remove_overlaps(sub_positions)
                        # compute the length of each subsentence
                        subsentence_len = len(subsent)
                        # check if sub_positions list is empty or not
                        if sub_positions:
                            # assign to sub_image_pointers the list of labels for the subsentence
                            # (it will be kept until a new subsentence.
                                # with labels won't be found)
                            temp = []
                            for pos in sub_positions:
                                temp = temp + pos[2]
                            sub_image_pointers = list(set(temp))
                            # classify labels in pre, post and in description (remember that
                            # there are no POS in this case).
                            # check if the last label extracted is at the end of the subsentence
                            if sub_positions[-1][1] == subsentence_len:
                            # consider labels as post description labels
                                # assign to the subcaptions the related sentences
                                subcaptions = post_labels(subcaptions, subsent, sub_positions)
                            # check if the first label extracted is at the beginning of the
                            # subsentence.
                            elif sub_positions[0][0] == 0:
                            # consider labels as pre description labels
                                # assign to the subcaptions the related subsentences
                                subcaptions = pre_labels(subcaptions, subsent, sub_positions)
                            # consider labels as 'fuzzy labels' and store the subsentence in
                            # fuzzycaptions.
                            else:
                                for key in sub_labels:
                                    try:
                                        fuzzycaptions[key] = fuzzycaptions[key] + subsent + ' '
                                    except:
                                        pass
                        # Assign the subsentence without labels to the subcaptions that have been
                        # expanded in the previous iteration.
                        else:
                            for label in sub_image_pointers:
                                try:
                                    subcaptions[label] = subcaptions[label] + subsent
                                except:
                                    pass

        # Assign the sentence without labels to the subcaptions that have been expanded in the
        # previous iteration.
        else:
            for label in image_pointers:
                subcaptions[label] = subcaptions[label] + subcapt
    # assign each element in fuzzycaptions to subcaptions, labelling it as 'fuzzy'
    for label, value in fuzzycaptions.items():
        subcaptions[label] = subcaptions[label] + ' FUZZY: ' + value
        #subcaptions[label] = value + ', ' + subcaptions[label]
    #print("subcaptions")
    #print (subcaptions)


    #print("ground_truth")
    #print(ground_truth_data[test_idx])

    for label_index, label in enumerate(selected_labels):
        print("label " + label)
        print("predicted " + label)
        print(subcaptions[label])
        print("ground_truth " + label)
        print(ground_truth_data[test_idx]['subcaptions'][label_index])
