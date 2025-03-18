import gzip
import argparse
from functools import partial

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from typing import List


def read_fasta(fasta_path: str) -> List[SeqRecord]:  # TODO Too large for memory?
    with open(fasta_path, "rb") as handle:
        gzipped = handle.read(2) == b'\x1f\x8b'

    _open = partial(gzip.open, mode="rt") if gzipped else open
    with _open(fasta_path) as handle:
        return list(SeqIO.parse(handle, "fasta"))  # Or should we yield?