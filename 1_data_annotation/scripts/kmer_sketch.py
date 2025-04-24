import argparse
from Bio import SeqIO
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

DEFAULT_K = 15
DEFAULT_SKETCH_SIZE = 2**14

def read_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences


def parse_args():
    parser = argparse.ArgumentParser(description="Generate kmer sketch from FASTA file.")
    parser.add_argument('--fasta', type=str, required=True, help="Path to the input FASTA file.")
    parser.add_argument('--output', type=str, required=True, help="Path to the output sketch file.")
    parser.add_argument('--k', type=int, default=6, help="Length of the kmer.")
    parser.add_argument('--sketch_size', type=int, default=16384, help="Size of the sketch.")
    return parser.parse_args()


def main(args):
    print("Reading FASTA file...")
    sequences = read_fasta(args.fasta)

    print(f"Generating kmer sketch with k={args.k} and sketch size={args.sketch_size}...")
    hv = HashingVectorizer(
        analyzer='char',
        ngram_range=(args.k, args.k),
        n_features=args.sketch_size,
        lowercase=False,
        alternate_sign=False,
    )
    X_sketch = hv.transform(sequences)
    X_sketch = sparse.csr_matrix(X_sketch)  # Just ensuring the output is in CSR format
    sparse.save_npz(args.output, X_sketch)
    print(f"Sketch saved to {args.output}.")


if __name__ == "__main__":
    main(parse_args())