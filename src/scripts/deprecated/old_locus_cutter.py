# Author: Sam Horsfield

import mappy as mp
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import os
import argparse
from collections import defaultdict
import gzip
from functools import partial
import tqdm

def get_options():
    description = "Cuts out loci based on alignment of reference sequences"
    parser = argparse.ArgumentParser(description=description,
                                     prog='python locus_cutter.py')
    IO = parser.add_argument_group('Input/options.out')
    IO.add_argument('--infiles',
                    required=True,
                    help='List of file paths to cut, one per line')
    IO.add_argument('--query',
                    help='Fasta file of sequences to align and cut. Each cut will be conducted with a pair of sequences paired using the same key.'
                         'Ensure sequences are on the positive strand and are placed in the file in the same order they appear on the positive strand'
                         'e.g. dexB -> aliA for S. pneumoniae CBL.')
    IO.add_argument('--cutoff',
                    type=float,
                    default=0.7,
                    help='Cutoff of alignment length to confirm match. '
                         'Default = 0.7')
    IO.add_argument('--outpref',
                    default="result_cut.fasta",
                    help='Output filename. Default = "result_cut.fasta"')
    return parser.parse_args()

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
            best_map = (pair_id, seq_index, hit.ctg, hit.r_st, hit.r_en, query_hit, hit.strand)

    return best_map

def cut_loci(infiles, seq_pair_dict, cutoff):
    file_list = []

    cut_records = []
    not_found = set()
    partial_found = set()

    with open(infiles, "r") as f:
        for line in f:
            file_list.append(line.strip())

    for file in tqdm.tqdm(file_list):
        # check if FASTA is gzipped
        gzipped = False
        with open(file, 'rb') as test_f:
            gzipped = True if test_f.read(2) == b'\x1f\x8b' else False

        _open = partial(gzip.open, mode='rt') if gzipped == True else open
        
        for pair_id, seq_pair in seq_pair_dict.items():
        
            best_map_pair = [None, None]
            # find best match for each sequence
            for seq_index, seq in enumerate(seq_pair):
                best_map_pair[seq_index] = get_best_map(file, pair_id, seq_index, seq, cutoff)

            seq1_valid = best_map_pair[0][0] != None
            seq2_valid = best_map_pair[1][0] != None

            # make sure at least one sequences found a hit
            if seq1_valid or seq2_valid:
                #print(best_map_pair[0])
                #print(best_map_pair[1])
                with _open(file) as handle:
                    fasta_sequences = SeqIO.parse(handle, 'fasta')
                    for fasta in fasta_sequences:
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
                            
                            strand_str = "_for" if (strand1 == 1 and strand2 == 1) else "_rev"
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
                                # positive strand, set locus_2 as end of contig
                                if strand == 1:
                                    locus_1 = best_map_pair[0][3]
                                    locus_2 = len(sequence) + 1
                                # negative strand, set locus_1 as beginning of contig
                                else:
                                    locus_1 = 0
                                    locus_2 = best_map_pair[0][4]
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
                                # positive strand, set locus_1 as start of contig
                                if strand == 1:
                                    locus_1 = 0
                                    locus_2 = best_map_pair[1][4]
                                # negative strand, set locus_2 as end of contig
                                else:
                                    locus_1 = best_map_pair[1][3]
                                    locus_2 = len(sequence) + 1
                                detail = "2_extended"
                            else:
                                locus_1 = best_map_pair[1][3]
                                locus_2 = best_map_pair[1][4]
                                detail = "2_only"
                        
                        # ensure match found
                        if locus_1 != None and locus_2 != None:                            
                            cut = sequence[locus_1:locus_2]

                            # get sequence onto correct strand
                            if strand_str == "_rev":
                                cut = str(Seq(cut).reverse_complement())
                                strand_str = "_for"
                            
                            detail += strand_str

                            cut_records.append(SeqRecord(Seq(cut), id=contig_id + "_" + pair_id + "_" + detail,
                                                        description=fasta.description))
            else:
                # add to list of genomes not found
                not_found.add(file)
    
    return cut_records, partial_found, not_found

def main():
    options = get_options()
    infiles = options.infiles
    query = options.query
    cutoff = options.cutoff
    outpref = options.outpref
    
    # check if FASTA is gzipped
    gzipped = False
    with open(query, 'rb') as test_f:
        gzipped = True if test_f.read(2) == b'\x1f\x8b' else False

    _open = partial(gzip.open, mode='rt') if gzipped == True else open
    
    # get pairs of query sequences
    seq_pair_dict = defaultdict(list)

    with _open(query) as handle:
        fasta_sequences = SeqIO.parse(handle, 'fasta')
        for fasta in fasta_sequences:
            id, sequence = fasta.id, str(fasta.seq)
            seq_pair_dict[id].append(sequence)
    
    # check that each sequence pair only has two sequences
    for pair_id, seq_pair in seq_pair_dict.items():
        assert len(seq_pair) == 2
    
    # generate list of cut loci
    cut_records, partial_found, not_found = cut_loci(infiles, seq_pair_dict, cutoff)

    # write cut loci
    SeqIO.write(cut_records, outpref + ".fasta", "fasta")

    # write partial_found and not found
    with open(outpref + "_partial.txt", "w") as o:
        for entry in partial_found:
            o.write(entry + "\n")
    
    with open(outpref + "_absent.txt", "w") as o:
        for entry in not_found:
            o.write(entry + "\n")

    return 0

if __name__ == "__main__":
    main()
