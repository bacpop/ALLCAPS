# Author: Sam Horsfield
# Modified by: Alireza Tajmirriahi
# Made adjustments to
# - run on GPS instead of AtB,
# - utilize multi-threading,
# - store non-CBL sequences for downstream tasks.

import mappy as mp
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import os
import gzip
import argparse
from tqdm import tqdm
import numpy as np
from functools import partial
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from .consts import CONTIG_SEP, NONCBL_ID_PREFIX, RND_STATE
from .logging_config import get_logger

logger = get_logger(__name__)


# Legacy separator. Record ids used to be `{public}__{contig}_{pair}_{detail}`, which
# no downstream module could parse: build_contig_metadata requires `Public_ID#Contig_ID`
# and raised ValueError on everything this script produced, so the assembly -> training
# route never ran. ID_SEP is now only used to recover a public name from a *filename*.
ID_SEP = "__"
DEFAULT_MAX_EXTENSION = 30_000
np.random.seed(RND_STATE)


def extract_public_name(file_name):
    basename = os.path.basename(file_name)
    if ID_SEP in basename:
        name = basename.split(ID_SEP)[0]
    else:
        name = basename.split(".")[0]
    if CONTIG_SEP in name:
        # A `#` in the sample name would make `Public_ID#Contig_ID` ambiguous.
        raise ValueError(
            f"Sample name {name!r} (from {basename!r}) contains {CONTIG_SEP!r}, which is "
            f"the reserved Public_ID/Contig_ID separator. Rename the input file."
        )
    return name


_UNSAFE_IN_PATH = str.maketrans({"/": "_", "\\": "_", " ": "_", "\t": "_", CONTIG_SEP: "_"})


def safe_contig_id(contig_id):
    """Make a contig name usable as both an id field and a filename.

    Downstream, each record becomes `<record id>.npy`, so a `/` in a contig name
    would be read as a directory. `#` is reserved as the Public_ID/Contig_ID
    separator. Dots are fine and are left alone — assembler names are full of them
    (`NODE_20_length_35242_cov_40.42_pilon`) and the readers use `splitext`.
    """
    return str(contig_id).translate(_UNSAFE_IN_PATH)


def cbl_record_id(public_name, contig_id, pair_id=None):
    """`Public_ID#Contig_ID` — the one format build_contig_metadata accepts.

    ``pair_id`` is appended to the contig field only when the query FASTA defines more
    than one flank pair, so the common single-pair case stays clean (``ERR1788086#7``).
    Provenance (which pair, which cut case, which strand) lives in the description.
    """
    contig = safe_contig_id(contig_id)
    if pair_id is not None:
        contig = f"{contig}.{safe_contig_id(pair_id)}"
    return f"{public_name}{CONTIG_SEP}{contig}"


def noncbl_record_id(public_name, contig_id, fragment=None, pair_id=None):
    """`NONCBL#Public_ID#Contig_ID`, keeping the prefix build_contig_metadata expects.

    ``fragment`` disambiguates the several non-capsular pieces one contig can yield —
    the sequence before the cut (``p``) and after it (``s``) — which would otherwise
    collide on a single id and be silently dropped by the embedder's skip-if-exists.
    """
    contig = safe_contig_id(contig_id)
    if pair_id is not None:
        contig = f"{contig}.{safe_contig_id(pair_id)}"
    if fragment:
        contig = f"{contig}{fragment}"
    return f"{NONCBL_ID_PREFIX}{CONTIG_SEP}{public_name}{CONTIG_SEP}{contig}"


def file_handler(file):
    """
    Check if the file is gzipped and return the appropriate open function.
    """
    with open(file, "rb") as test_f:
        gzipped = test_f.read(2) == b"\x1f\x8b"
    return partial(gzip.open, mode="rt") if gzipped else open


def build_parser():
    description = "Cuts out loci based on alignment of reference sequences"
    parser = argparse.ArgumentParser(
        description=description, prog="python locus_cutter.py"
    )
    IO = parser.add_argument_group("Input/options.out")
    IO.add_argument(
        "--infiles", required=True, help="List of file paths to cut, one per line."
    )
    IO.add_argument(
        "--query",
        help="Fasta file of sequences to align and cut. Each cut will be conducted with a pair of sequences paired using the same key. "
        "Ensure sequences are on the positive strand and are placed in the file in the same order they appear on the positive strand "
        "e.g. dexB -> aliA for S. pneumoniae CBL.",
    )
    IO.add_argument(
        "--cutoff",
        type=float,
        default=0.7,
        help="Cutoff of alignment length to confirm match. Default = 0.7",
    )
    IO.add_argument(
        "--outpref",
        default="result_cut.fasta",
        help='Output filename. Default = "result_cut.fasta"',
    )
    IO.add_argument(
        "--max-extension",
        type=int,
        default=DEFAULT_MAX_EXTENSION,
        help="Maximum number of bases to extend toward contig ends when only one flank hits. Default = 30000",
    )
    IO.add_argument(
        "--save-noncbl",
        action="store_true",
        help="Save non-CBL sequences to a separate file. Default = False",
    )
    return parser


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    return args


def get_best_map(index, pair_id, seq_index, sequence, cutoff):
    a = mp.Aligner(index, preset="asm10")

    best_map = (None, None, None, 0, 0, 0, 0)

    for hit in a.map(sequence):
        if not hit.is_primary:
            continue
        query_hit = hit.blen

        # set cutoff for minimum alignment length
        if query_hit < cutoff * len(sequence):
            continue

        if query_hit > best_map[-1]:
            best_map = (
                pair_id,
                seq_index,
                hit.ctg,
                hit.r_st,
                hit.r_en,
                query_hit,
                hit.strand,
            )

    return best_map


def assert_unique_ids(records, what):
    """Fail loudly on duplicate record ids rather than losing records downstream."""
    seen = {}
    for record in records:
        seen[record.id] = seen.get(record.id, 0) + 1
    duplicates = {rid: n for rid, n in seen.items() if n > 1}
    if duplicates:
        sample = ", ".join(f"{rid} (x{n})" for rid, n in list(duplicates.items())[:5])
        raise ValueError(
            f"{len(duplicates)} duplicate record id(s) among {what}: {sample}"
            + ("..." if len(duplicates) > 5 else "")
        )


def save_non_cbl(records, missing_files, list_len, out_path):
    """
    Save non-CBL records to a separate file. The missing files contigs are also appended.
    Too short sequences are filtered out. Too long sequences are randomly subsampled.
    """
    MIN_LENGTH, MAX_LENGTH = 5000, 25000  # TODO use the histogram from
    for missing_file in missing_files:
        _open = file_handler(missing_file)
        public_name = extract_public_name(missing_file)
        with _open(missing_file) as f:
            fasta_sequences = SeqIO.parse(f, "fasta")
            for fasta in fasta_sequences:
                contig_id, sequence = fasta.id, str(fasta.seq)
                records.append(
                    SeqRecord(
                        Seq(sequence),
                        id=noncbl_record_id(public_name, contig_id),
                        description=f"noncbl_missing {fasta.description}",
                    )
                )

    # Filter out too short sequences
    records = [record for record in records if len(record.seq) >= MIN_LENGTH]
    # Subsample too long sequences according to the list_len distribution
    records = list(
        map(
            lambda record: (
                record
                if len(record.seq) <= MAX_LENGTH
                else SeqRecord(
                    Seq(str(record.seq)[: min(MAX_LENGTH, np.random.choice(list_len))]),
                    # Same record, shortened — keep the id, note it in the description.
                    id=record.id,
                    description=f"subsampled {record.description}",
                )
            ),
            records,
        )
    )
    assert_unique_ids(records, "non-CBL fragments")
    with open(out_path, "w") as o:
        SeqIO.write(records, o, "fasta")


def process_file(file, seq_pair_dict, cutoff, max_extension):
    cut_records = []
    partial_found = set()
    not_found = set()
    non_cbl_records = []

    _open = file_handler(file)

    # Only disambiguate ids by flank pair when there is more than one pair to tell
    # apart; with the single shipped dexB/aliA pair, ids stay plain `ERR1788086#7`.
    multi_pair = len(seq_pair_dict) > 1

    # For each pair in the query dictionary, do the mapping
    for pair_id, seq_pair in seq_pair_dict.items():
        id_pair_id = pair_id if multi_pair else None
        best_map_pair = [None, None]
        for seq_index, seq in enumerate(seq_pair):
            best_map_pair[seq_index] = get_best_map(
                file, pair_id, seq_index, seq, cutoff
            )

        # If at least one hit is found, open the file and extract loci.
        seq1_valid = best_map_pair[0][0] is not None
        seq2_valid = best_map_pair[1][0] is not None
        if seq1_valid or seq2_valid:
            with _open(file) as handle:
                fasta_sequences = SeqIO.parse(handle, "fasta")
                public_name = extract_public_name(file)
                for fasta in fasta_sequences:
                    # Append each valid SeqRecord to cut_records.
                    contig_id, sequence = fasta.id, str(fasta.seq)
                    seq1_match = contig_id == best_map_pair[0][2]
                    seq2_match = contig_id == best_map_pair[1][2]
                    locus_1 = None
                    locus_2 = None
                    detail = ""
                    strand_str = None

                    # both sequences match contig
                    if seq1_match and seq2_match:
                        # work out which way round to cut
                        strand1 = best_map_pair[0][-1]
                        strand2 = best_map_pair[1][-1]
                        locus_1 = min(best_map_pair[0][3], best_map_pair[1][3])
                        locus_2 = max(best_map_pair[0][4], best_map_pair[1][4])

                        strand_str = (
                            "_for" if (strand1 == 1 and strand2 == 1) else "_rev"
                        )
                        detail = "complete"

                    # first sequence matches
                    elif seq1_match and not seq2_match:
                        # add partial match
                        partial_found.add(file)

                        # get strand
                        strand = best_map_pair[0][-1]
                        strand_str = "_for" if strand == 1 else "_rev"

                        # if seq2_valid, means likely contig break
                        if seq2_valid:
                            # positive strand, extend towards contig end but keep total length <= max_extension
                            if strand == 1:
                                locus_1 = best_map_pair[0][3]
                                locus_2 = min(len(sequence), locus_1 + max_extension)
                            # negative strand, extend towards contig start but keep total length <= max_extension
                            else:
                                locus_2 = best_map_pair[0][4]
                                locus_1 = max(0, locus_2 - max_extension)
                            detail = "1_extended"
                        else:
                            locus_1 = best_map_pair[0][3]
                            locus_2 = best_map_pair[0][4]
                            detail = "1_only"
                    # second sequence matches
                    elif not seq1_match and seq2_match:
                        # add partial match
                        partial_found.add(file)

                        # get strand
                        strand = best_map_pair[1][-1]
                        strand_str = "_for" if strand == 1 else "_rev"

                        # if seq1_valid, means likely contig break
                        if seq1_valid:
                            # positive strand, extend towards contig start but keep total length <= max_extension
                            if strand == 1:
                                locus_2 = best_map_pair[1][4]
                                locus_1 = max(0, locus_2 - max_extension)
                            # negative strand, extend towards contig end but keep total length <= max_extension
                            else:
                                locus_1 = best_map_pair[1][3]
                                locus_2 = min(len(sequence), locus_1 + max_extension)
                            detail = "2_extended"
                        else:
                            locus_1 = best_map_pair[1][3]
                            locus_2 = best_map_pair[1][4]
                            detail = "2_only"

                    # ensure match found
                    if locus_1 != None and locus_2 != None:
                        cut = sequence[locus_1:locus_2]
                        pref, suff = sequence[:locus_1], sequence[locus_2:]
                        if pref:
                            pref = (
                                str(Seq(pref).reverse_complement())
                                if strand_str == "_rev"
                                else pref
                            )
                            non_cbl_records.append(
                                SeqRecord(
                                    Seq(pref),
                                    id=noncbl_record_id(
                                        public_name, contig_id, "p", id_pair_id
                                    ),
                                    description=f"noncbl_prefix pair={pair_id} "
                                    f"{fasta.description}",
                                )
                            )
                        if suff:
                            suff = (
                                str(Seq(suff).reverse_complement())
                                if strand_str == "_rev"
                                else suff
                            )
                            non_cbl_records.append(
                                SeqRecord(
                                    Seq(suff),
                                    id=noncbl_record_id(
                                        public_name, contig_id, "s", id_pair_id
                                    ),
                                    description=f"noncbl_suffix pair={pair_id} "
                                    f"{fasta.description}",
                                )
                            )
                        # get sequence onto correct strand
                        if strand_str == "_rev":
                            cut = str(Seq(cut).reverse_complement())
                            strand_str = "_for"

                        detail += strand_str
                        cut_records.append(
                            SeqRecord(
                                Seq(cut),
                                id=cbl_record_id(public_name, contig_id, id_pair_id),
                                # The cut case and strand used to live in the record id,
                                # where nothing downstream could parse around them.
                                description=f"cut={detail} pair={pair_id} "
                                f"{fasta.description}",
                            )
                        )
        else:
            not_found.add(file)

    return cut_records, partial_found, not_found, non_cbl_records


def parallel_cut_loci(file_list, seq_pair_dict, cutoff, max_extension):
    all_cut_records = []
    all_partial_found = set()
    all_not_found = set()
    all_non_cbl_records = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_file, file, seq_pair_dict, cutoff, max_extension)
            for file in file_list
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            cut_records, partial_found, not_found, non_cbl_records = future.result()
            all_cut_records.extend(cut_records)
            all_partial_found.update(partial_found)
            all_not_found.update(not_found)
            all_non_cbl_records.extend(non_cbl_records)
    return all_cut_records, all_partial_found, all_not_found, all_non_cbl_records


def main(options):
    infiles = options.infiles
    query = options.query
    cutoff = options.cutoff
    outpref = options.outpref

    # check if FASTA is gzipped
    gzipped = False
    with open(query, "rb") as test_f:
        gzipped = True if test_f.read(2) == b"\x1f\x8b" else False

    _open = partial(gzip.open, mode="rt") if gzipped == True else open

    # get pairs of query sequences
    seq_pair_dict = defaultdict(list)

    logger.info("Reading query sequences from %s...", query)
    with _open(query) as handle:
        fasta_sequences = SeqIO.parse(handle, "fasta")
        for fasta in fasta_sequences:
            id, sequence = fasta.id, str(fasta.seq)
            seq_pair_dict[id].append(sequence)

    # check that each sequence pair only has two sequences
    for _, seq_pair in seq_pair_dict.items():
        assert len(seq_pair) == 2

    logger.info("Processing files listed in %s...", infiles)
    with open(infiles, "r") as f:
        file_list = [line.strip() for line in f.readlines()]

    cut_records, partial_found, not_found, non_cbl_records = parallel_cut_loci(
        file_list, seq_pair_dict, cutoff, options.max_extension
    )

    logger.info("Writing cut loci...")
    out_dir = os.path.dirname(outpref)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Duplicate ids are silent data loss downstream: embed_transformer writes one
    # `<record id>.npy` per record and skips paths that already exist, so a collision
    # drops every record after the first.
    assert_unique_ids(cut_records, "cut loci")

    SeqIO.write(cut_records, outpref + ".fasta", "fasta")

    logger.info("Writing partial and not found files...")
    with open(outpref + "_partial.txt", "w") as o:
        for entry in partial_found:
            o.write(entry + "\n")
    with open(outpref + "_absent.txt", "w") as o:
        for entry in not_found:
            o.write(entry + "\n")

    if options.save_noncbl:
        logger.info("Saving non-CBL sequences...")
        save_non_cbl(
            non_cbl_records,
            not_found,
            list(map(len, cut_records)),
            outpref + "_noncbl.fasta",
        )
    logger.info("Cut loci written to %s*", outpref)


if __name__ == "__main__":
    main(parse_args())
