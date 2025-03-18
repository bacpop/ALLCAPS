import gzip
import argparse
from functools import partial

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from typing import List


DEFAULT_CBL_PATH = "/nfs/research/jlees/shorsfield/RARA_pneumo_cps/AtB_All_S_pneumoniae_CBL.fasta"
DEFAULT_META_PATH = "/hps/software/users/jlees/tajmirri/sandbox/ena_metadata_compressed_known_serotype.tsv.gz"


def read_fasta(fasta_path: str) -> List[SeqRecord]:  # TODO Too large for memory?
    with open(fasta_path, "rb") as handle:
        gzipped = handle.read(2) == b'\x1f\x8b'

    _open = partial(gzip.open, mode="rt") if gzipped else open
    with _open(fasta_path) as handle:
        return list(SeqIO.parse(handle, "fasta"))  # Or should we yield?


def get_sample_accession(record: SeqRecord) -> str:
    return record.id.split(".")[0]


def main(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cbl", default=DEFAULT_CBL_PATH, help="Path to CBL fasta file")
    parser.add_argument("--meta", default=DEFAULT_META_PATH, help="Path to metadata file")
    main(parser.parse_args())