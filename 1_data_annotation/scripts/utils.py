import re


# A mapping of serotype to a more coarse label, consisting of
# similar groupings, i.e. genogroups or serogroups. We use it
# for contrastive training and visualization purposes.
SEROTYPE_GROUPS = {
    "6A": "6",
    "6B": "6",
    "6C": "6",
    "6D": "6",
    "6E(6B)": "6",
    "7B": "7B_7C_40",
    "7C": "7B_7C_40",
    "7F": "7A_7F",
    "9A": "9",
    "9L": "9",
    "9N": "9",
    "9V": "9",
    "10A": "10",
    "10B": "10",
    "10C": "10",
    "10F": "10",
    "10X": "33G",
    "11A": "11",
    "11B": "11",
    "11C": "11",
    "11E": "11",
    "12A": "12_44_46",
    "12B": "12_44_46",
    "12F": "12_44_46",
    "15A": "15",
    "15B": "15",
    "15C": "15",
    "15B/15C": "15",
    "15F": "15",
    "18A": "18",
    "18B": "18",
    "18C": "18",
    "18F": "18",
    "19A": "19A",
    "19B": "19B_19C",
    "19F": "19F",
    "20": "20",
    "20A": "20",
    "20B": "20",
    "22A": "22",
    "22F": "22",
    "23A": "23",
    "23B": "23",
    "23B1": "23",
    "23F": "23",
    "24": "24",
    "24A": "24",
    "24F": "24",
    "25A": "25A_25F_38",
    "25F": "25A_25F_38",
    "28A": "28",
    "28F": "28",
    "33A": "33A_33F_37",
    "33A/33F": "33A_33F_37",
    "33B": "33B_33D",
    "33D": "33B_33D",
    "33F": "33A_33F_37",
    "35A": "35A_35C_42",
    "35B": "35B_35D",
    "35B/35D": "35B_35D",
    "35C": "35A_35C_42",
    "35D": "35B_35D",
    "37": "33A_33F_37",
    "38": "25A_25F_38",
    "39X": "10D",
    "40": "7B_7C_40",
    "42": "35A_35C_42",
    "46": "12_44_46"
}


def map_serotype_to_group(serotype):
    """ Map serotype to a more coarse label by
    looking it up in the serogroup/genogroups data. """
    if isinstance(serotype, str):
        return SEROTYPE_GROUPS.get(serotype, serotype)
    print(f"Warning: {serotype} not found in mapping.")
    return serotype


def extract_serogroup(serotype):
    """ Map serotype to a more coarse label by
    extracting the number from the serotype string. """
    if isinstance(serotype, str):
        match = re.search(r"\d+", serotype)
        if match:
            return str(match.group())
    return serotype
