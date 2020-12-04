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
Useful regular expressions for caption splitting.
"""

import re

from typing import Pattern

from ..utils.figure.label import labels_structure as ls

_UC_ROMAN: str = r'[' + ','.join(ls.UC_ROMAN) + r']'
_LC_ROMAN: str = r'[' + ','.join(ls.LC_ROMAN) + r']'

# = '[i,ii,iii,iv...,xx]|[I,II,II,IV,...,XX]'
_ANY_ROMAN: str = _LC_ROMAN + r'+|' + _UC_ROMAN + r'+'
_ANY_CHAR: str = r"[a-z]|[A-Z]"
_ANY_NON_ZERO_DIGIT: str = r"[1-9]"

_ANY_LABEL: str = _ANY_CHAR + r'|' + _ANY_ROMAN + r'|' + _ANY_NON_ZERO_DIGIT

# Build the regular expressions that match classes.
RE_CHARACTERS: Pattern[str] = re.compile(r"((\b|\.|\()(" + _ANY_CHAR \
                                       + r")(?![^).:0-9])[0-9]?(\)|\.|:|\b)?)")

RE_ROMAN: Pattern[str] = re.compile(r"((\b|\.|\()(" + _ANY_ROMAN \
                                  + r")+(?![^).:])(\)|\.|:|\b)?)")

RE_DIGITS: Pattern[str] = re.compile(r"((\b|\.|\()(" + _ANY_NON_ZERO_DIGIT \
                                   + r")(?![^).:0-9])[0-9]?(\)|\.|:|\b)?)")

# Build the regular expressions that match ranges (hyphens and conjunctions).
RE_HYPHEN: Pattern[str] = re.compile(r"((\b|\.|\()(" + _ANY_LABEL \
                                   + r")[0-9]?-(" + _ANY_LABEL \
                                   + r")[0-9]?\)?)\b")

RE_CONJUNCTIONS: Pattern[str] = re.compile(
    r"((\b|\.|\()((" + _ANY_LABEL \
  + r")[0-9]?(\)|\.|:)?)((\s?,\s?\(?(" + _ANY_LABEL \
  + r")[0-9]?(\)|\.|:)?)+|(\s?and\s?\(?(" + _ANY_LABEL \
  + r")[0-9]?(\)|\.|:)?\W)+)(\s?,?\s?and\s?\(?(" + _ANY_LABEL \
  + r")[0-9]?(\)|\.|:)?\W)?\W\)?)")

# Define the regular expressions that match Part Of Speech (POS) labels.
# => single labels
RE_CHARACTERS_POS: Pattern[str] = re.compile(r"((in|from|panel(s)?)\s?\(?(" + _ANY_CHAR \
                                           + r")(?![^).:0-9])[0-9]?(\)|\.|:|\b)?)")

RE_ROMAN_POS: Pattern[str] = re.compile(r"((in|from|panel(s)?)\s?\(?("  + _ANY_ROMAN \
                                      + r")+(?![^).:])(\)|\.|:|\b)?)")

RE_DIGITS_POS: Pattern[str] = re.compile(r"((in|from|panel(s)?)\s?\(?(" + _ANY_NON_ZERO_DIGIT \
                                       + r")(?![^).:0-9])[0-9]?(\)|\.|:|\b)?)")

# => range labels
RE_HYPHEN_POS: Pattern[str] = re.compile(r"((in|from|panel(s)?)\s?\(?(" + _ANY_LABEL \
                                       + r")[0-9]?-(" + _ANY_LABEL \
                                       + r")[0-9]?\)?)")

RE_CONJUNCTIONS_POS: Pattern[str] = re.compile(
      r"((in|from|panel(s)?)\s?\(?((" + _ANY_LABEL \
    + r")[0-9]?(\)|\.|:)?)((\s?,\s?\(?(" + _ANY_LABEL \
    + r")[0-9]?(\)|\.|:)?)+|(\s?and\s?\(?(" + _ANY_LABEL \
    + r")[0-9]?(\)|\.|:)?\W)+)(\s?,?\s?and\s?\(?(" + _ANY_LABEL \
    + r")[0-9]?(\)|\.|:)?\W)?\W\)?)")
