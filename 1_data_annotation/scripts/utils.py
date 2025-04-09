import re


def map_serotype_to_group(serotype):
    """ Extract the number from the serotype string. """
    if isinstance(serotype, str):
        match = re.search(r"\d+", serotype)
        if match:
            return str(match.group())
    return serotype
