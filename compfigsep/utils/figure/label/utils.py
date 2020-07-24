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


#########################
Utilities for label text.
"""

LC_ROMAN = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
            'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx']

# {'i': 1, 'ii': 2,..., 'xx': 20}
LC_ROMAN_TO_INT = dict(zip(LC_ROMAN,
                           range(1, len(LC_ROMAN) + 1)))

# {1: 'i', 2: 'ii',..., 20: 'xx'}
LC_ROMAN_FROM_INT = {value: char for char, value in LC_ROMAN_TO_INT.items()}

# ['I', 'II',..., 'XX']
UC_ROMAN = [char.upper() for char in LC_ROMAN]

# {'I': 1, 'II': 2,..., 'XX': 20}
UC_ROMAN_TO_INT = dict(zip(UC_ROMAN,
                           range(1, len(UC_ROMAN) + 1)))


# {1: 'I', 2: 'II',..., 20: 'XX'}
UC_ROMAN_FROM_INT = {value: char for char, value in UC_ROMAN_TO_INT.items()}


def is_upper_char(char: str) -> bool:
    """
    Test whether the given character is an upper alphabetical char.

    Args:
        char (str): A character.

    Returns:
        is_upper (bool):    Whether the character is an upper alphabetical char.
    """
    if len(char) != 1:
        return False

    return ord(char) in range(65, 65 + 26)


def is_lower_char(char: str) -> bool:
    """
    Test whether the given character is a lower alphabetical char.

    Args:
        char (str): A character.

    Returns:
        is_lower (bool):    Whether the character is a lower alphabetical char.
    """
    if len(char) != 1:
        return False

    return ord(char) in range(97, 97 + 26)


def is_char(char: str) -> bool:
    """
    Test whether the given character is an alphabetical char.

    Args:
        char (str): A character.

    Returns:
        is_char (bool): Whether the character is an alphabetical char.
    """
    if len(char) != 1:
        return False

    return is_upper_char(char) or is_lower_char(char)


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


def roman_to_int(roman_char: str) -> int:
    """
    Convert a character representing a roman number (upper or lower case) to its numerical value.

    Args:
        roman_char (str):   A character representing a roman number.

    Returns:
        (int):  The numerical value of the roman number.
    """
    assert roman_char in UC_ROMAN or roman_char in LC_ROMAN,\
        f"TODO {roman_char}"

    if roman_char in UC_ROMAN:
        return UC_ROMAN_TO_INT[roman_char]

    return LC_ROMAN_TO_INT[roman_char]
