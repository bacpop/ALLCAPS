"""Reading the user's list of input FASTA files.

Both ``predict`` and ``train`` take a manifest. ``predict`` needs only paths; ``train``
additionally needs a serotype per sample. Accepted shapes, so that nobody has to guess:

* a plain text file, one FASTA path per line (``#`` comments and blanks ignored)
* a CSV or TSV with a header, containing a path column and -- for training -- a
  serotype column
* bare paths given straight on the command line

Sample ids come from :func:`allcaps.data_locus_cutter.extract_public_name`, the same
function the cutter uses, so the ids in a manifest always line up with the record ids in
the cut FASTA.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from ..data_locus_cutter import extract_public_name
from ..logging_config import get_logger

logger = get_logger(__name__)

#: Column names accepted for the FASTA path, in order of preference.
PATH_COLUMNS = ("path", "fasta", "file", "filename", "assembly", "filepath")
#: Column names accepted for the serotype label.
SEROTYPE_COLUMNS = ("serotype", "in_silico_serotype", "label")
#: Column names accepted for an explicit sample id.
ID_COLUMNS = ("sample", "sample_id", "public_id", "id", "err")

FASTA_SUFFIXES = {".fa", ".fas", ".fasta", ".fna", ".ffn", ".contigs"}


@dataclass
class Sample:
    """One input assembly."""

    path: Path
    sample_id: str
    serotype: Optional[str] = None


def _looks_tabular(path: Path) -> bool:
    """True when the file has a delimiter-separated header rather than bare paths."""
    with open(path, newline="") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # A header row names columns; a path row is a path. Treat a first
            # non-comment line containing a known column name as tabular.
            lowered = [c.strip().lower() for c in line.replace("\t", ",").split(",")]
            return any(c in PATH_COLUMNS for c in lowered)
    return False


def _sniff_dialect(path: Path) -> str:
    with open(path, newline="") as fh:
        head = fh.readline()
    return "\t" if head.count("\t") >= head.count(",") else ","


def _pick(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {(f or "").strip().lower(): f for f in fieldnames}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _resolve(raw: str, relative_to: Path) -> Path:
    """Resolve a manifest entry, allowing paths relative to the manifest itself."""
    p = Path(raw).expanduser()
    if p.is_absolute() or p.exists():
        return p
    sibling = (relative_to.parent / p).expanduser()
    return sibling if sibling.exists() else p


def read_manifest(
    source: Path,
    require_serotype: bool = False,
) -> List[Sample]:
    """Parse a manifest file into :class:`Sample` rows."""
    source = Path(source).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Manifest not found: {source}")

    samples: List[Sample] = []

    if _looks_tabular(source):
        delimiter = _sniff_dialect(source)
        with open(source, newline="") as fh:
            rows = [r for r in csv.DictReader(fh, delimiter=delimiter)]
        if not rows:
            raise ValueError(f"Manifest {source} has a header but no rows.")
        fields = list(rows[0].keys())
        path_col = _pick(fields, PATH_COLUMNS)
        if path_col is None:
            raise ValueError(
                f"{source} has no recognised path column. Expected one of: "
                f"{', '.join(PATH_COLUMNS)}. Found: {', '.join(str(f) for f in fields)}"
            )
        sero_col = _pick(fields, SEROTYPE_COLUMNS)
        id_col = _pick(fields, ID_COLUMNS)
        if require_serotype and sero_col is None:
            raise ValueError(
                f"{source} has no serotype column, which `ALLCAPS train` needs. "
                f"Expected one of: {', '.join(SEROTYPE_COLUMNS)}.\n"
                f"Example manifest:\n    path,serotype\n"
                f"    /data/ERR1788086.fasta,19A\n    /data/ERR714669.fasta,3"
            )
        for n, row in enumerate(rows, start=2):
            raw = (row.get(path_col) or "").strip()
            if not raw:
                logger.warning("%s line %d: empty path, skipping.", source, n)
                continue
            path = _resolve(raw, source)
            serotype = (row.get(sero_col) or "").strip() if sero_col else None
            sample_id = (row.get(id_col) or "").strip() if id_col else ""
            samples.append(
                Sample(
                    path=path,
                    sample_id=sample_id or extract_public_name(str(path)),
                    serotype=serotype or None,
                )
            )
    else:
        if require_serotype:
            raise ValueError(
                f"{source} looks like a plain list of paths, but `ALLCAPS train` needs a "
                f"serotype for each sample. Supply a CSV instead:\n"
                f"    path,serotype\n    /data/ERR1788086.fasta,19A"
            )
        with open(source) as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                path = _resolve(line, source)
                samples.append(
                    Sample(path=path, sample_id=extract_public_name(str(path)))
                )

    return _validate(samples, require_serotype)


def from_paths(paths: Sequence[str]) -> List[Sample]:
    """Build samples from bare command-line paths."""
    samples = [
        Sample(path=Path(p).expanduser(), sample_id=extract_public_name(str(p)))
        for p in paths
    ]
    return _validate(samples, require_serotype=False)


def _validate(samples: List[Sample], require_serotype: bool) -> List[Sample]:
    if not samples:
        raise ValueError("No input FASTA files found.")

    missing = [s.path for s in samples if not s.path.is_file()]
    if missing:
        shown = "\n  ".join(str(m) for m in missing[:10])
        more = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(f"{len(missing)} input file(s) do not exist:\n  {shown}{more}")

    if require_serotype:
        unlabelled = [s.sample_id for s in samples if not s.serotype]
        if unlabelled:
            raise ValueError(
                f"{len(unlabelled)} sample(s) have no serotype: "
                f"{', '.join(unlabelled[:5])}{'...' if len(unlabelled) > 5 else ''}"
            )

    # Sample ids become Public_ID, which keys everything downstream. Two input files
    # collapsing onto one id would silently merge two assemblies.
    seen: dict = {}
    for s in samples:
        seen.setdefault(s.sample_id, []).append(s.path)
    clashes = {k: v for k, v in seen.items() if len(v) > 1}
    if clashes:
        detail = "\n  ".join(
            f"{k}: " + ", ".join(str(p) for p in v) for k, v in list(clashes.items())[:5]
        )
        raise ValueError(
            f"{len(clashes)} sample id(s) are claimed by more than one input file. Ids "
            f"come from the filename, so rename the files or add an explicit id column."
            f"\n  {detail}"
        )

    odd = [s.path.name for s in samples if s.path.suffix.lower() not in FASTA_SUFFIXES
           and not s.path.name.lower().endswith((".gz", ".fasta.gz", ".fa.gz"))]
    if odd:
        logger.warning(
            "%d input file(s) have an unusual FASTA extension (%s). Proceeding anyway.",
            len(odd), ", ".join(odd[:3]),
        )

    logger.info("Read %d input sample(s).", len(samples))
    return samples
