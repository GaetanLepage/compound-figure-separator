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


###################################################################################
Constants and function to deal with characters of caption labels and their classes.
"""

LABEL_CLASS_MAPPING = {
    '1': 0,
    '2': 1,
    '3': 2,
    '4': 3,
    '5': 4,
    '6': 5,
    '7': 6,
    '8': 7,
    '9': 8,
    'A': 9,
    'B': 10,
    'C': 11,
    'D': 12,
    'E': 13,
    'F': 14,
    'G': 15,
    'H': 16,
    'I': 17,
    'J': 18,
    'K': 19,
    'L': 20,
    'M': 21,
    'N': 22,
    'O': 23,
    'P': 24,
    'Q': 25,
    'R': 26,
    'S': 27,
    'T': 28,
    'U': 29,
    'V': 30,
    'W': 31,
    'X': 32,
    'Y': 33,
    'Z': 34,
    'a': 35,
    'b': 36,
    'd': 37,
    'e': 38,
    'f': 39,
    'g': 40,
    'h': 41,
    'i': 42,
    'j': 43,
    'l': 44,
    'm': 45,
    'n': 46,
    'q': 47,
    'r': 48,
    't': 49,
}
CLASS_LABEL_MAPPING = {v: k for k, v in LABEL_CLASS_MAPPING.items()}


def map_label(char: str) -> str:
    """
    Map the provided character to the right class.

    Args:
        char (str): A character [0-9][a-z][A-Z].

    Returns:
        char_class (str):   The class to which belongs the character.
    """
    assert len(char) == 1, f"Char should be of length 1: {char}"

    char_classes = [
        ('c', 'C'),
        ('k', 'K'),
        ('o', 'O'),
        ('p', 'P'),
        ('s', 'S'),
        ('u', 'U'),
        ('v', 'V'),
        ('w', 'W'),
        ('x', 'X'),
        ('y', 'Y'),
        ('z', 'Z')]

    # If the characetr is from a two characters class, return the upper-case one
    for char_class in char_classes:
        if char in char_class:
            return char_class[1]

    # If the character is from a single character class, we return it
    return char

def case_same_label(char: str) -> bool:
    """
    Check if `char` belongs to a "dual class".

    Args:
        char (str): A character [0-9][a-z][A-Z].

    Returns:
        bool which tells whether `char` belongs to a "dual class".
    """
    # If the given character belongs to a "dual class", return True.
    if char in (
            'c', 'C',
            'k', 'K',
            'o', 'O',
            'p', 'P',
            's', 'S',
            'u', 'U',
            'v', 'V',
            'w', 'W',
            'x', 'X',
            'y', 'Y',
            'z', 'Z'):
        return True

    # Else, return False.
    return False
