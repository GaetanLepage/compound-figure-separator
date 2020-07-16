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


#################################################
Useful regular expressions for caption splitting.
"""

import re

from ..utils.figure import labels_structure as ls

UC_ROMAN = r'[' + ','.join(ls.UC_ROMAN) + r']'
LC_ROMAN = r'[' + ','.join(ls.LC_ROMAN) + r']'

# = '[i,ii,iii,iv...,xx]|[I,II,II,IV,...,XX]'
ANY_ROMAN =  LC_ROMAN + r'|' + UC_ROMAN
ANY_CHAR = r"[a-z][A-Z]"
ANY_NON_ZERO_DIGIT = r"[1-9]"

ANY_LABEL = ANY_CHAR + r'|' + ANY_ROMAN + r'|' + ANY_NON_ZERO_DIGIT

RE_CHARACTERS = re.compile(r"((\b|\.|\()(" + ANY_CHAR + r")(?![^).:0-9])[0-9]?(\)|\.|:|\b)?)")
RE_ROMAN = re.compile(r"((\b|\.|\()(" + ANY_ROMAN + r")(?![^).:])(\)|\.|:|\b)?)")
RE_DIGITS = re.compile(r"((\b|\.|\()(" + ANY_NON_ZERO_DIGIT + r")(?![^).:0-9])[0-9]?(\)|\.|:|\b)?)")

# Build the regular expressions that match ranges (hyphens and conjunctions).
RE_HYPHEN = re.compile(r"((\b|\.|\()(" + ANY_LABEL + ")[0-9]?-([" + ANY_LABEL + r")[0-9]?\)?)")
RE_CONJUNCTIONS = re.compile(r"((\b|\.|\()((" + ANY_LABEL + r")[0-9]?(\)|\.|:)?)((\s?,\s?\(?(" + ANY_LABEL + r")[0-9]?(\)|\.|:)?)+|(\s?and\s?\(?(" + ANY_LABEL + r")[0-9]?(\)|\.|:)?\W)+)(\s?,?\s?and\s?\(?(" + ANY_LABEL + r")[0-9]?(\)|\.|:)?\W)?\W\)?)")

# Define the regular expressions that match Part Of Speech (POS) labels.
# => single labels
RE_CHARACTERS_POS = re.compile(r"((in|from|panel(s)?)\s?\(?(" + ANY_CHAR + r")(?![^).:0-9])[0-9]?(\)|\.|:|\b)?)")
RE_ROMAN_POS = re.compile(r"     ((in|from|panel(s)?)\s?\(?("      + ANY_ROMAN +    r")(?![^).:])(\)|\.|:|\b)?)")
RE_DIGITS_POS = re.compile(r"    ((in|from|panel(s)?)\s?\(?([1-9])            (?![^).:0-9])[0-9]?(\)|\.|:|\b)?)")
# => range labels
RE_HYPHEN_POS = re.compile(r"((in|from|panel(s)?)\s?\(?(" + ANY_LABEL + r")[0-9]?-([" + ANY_LABEL + r")[0-9]?\)?)")

RE_CONJUNCTIONS_POS = re.compile(r"((in|from|panel(s)?)\s?\(?((" + ANY_LABEL + r")[0-9]?(\)|\.|:)?)((\s?,\s?\(?(" + ANY_LABEL + r")[0-9]?(\)|\.|:)?)+|(\s?and\s?\(?(" + ANY_LABEL + r")[0-9]?(\)|\.|:)?\W)+)(\s?,?\s?and\s?\(?(" + ANY_LABEL + r")[0-9]?(\)|\.|:)?\W)?\W\)?)")
