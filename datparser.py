"""Utility for parsing a .dat file in the project specified format
Provides a function to parse fractions as well as dat files
Author: Daniel Chapin
"""


def parse_fraction(string) -> float:
    """Converts a string fraction into a float"""

    parts = [float(x) for x in string.split("/")]

    if len(parts) == 1:
        return parts[0]

    return parts[0] / parts[1]


def parse_dat(path) -> (list, list):
    """Parses a given .dat file
    returns a tuple containing a list of xvals and a list of yvals if successful
    returns None if the given file could not be opened

    Example usage:

        from datparser import parse_dat

        result = parse_dat("classA0.dat")

        if result != None:
            xvals, yvals = result
    """

    try:
        with open(path) as file:
            lines = file.readlines()

            xvals = [parse_fraction(x) for x in lines[0].split('\t')]
            yvals = [float(x) for x in lines[1].split('\t')]

            return (xvals, yvals)
    except:
        return None

    return None
