#!/usr/bin/env python3
"""Run ALLCAPS without installing it.

`pip install .` puts an `ALLCAPS` command on your PATH, which is the better route. This
shim exists so a plain clone works too:

    python ALLCAPS.py predict --input samples.txt --extract align --output results/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from allcaps.cli.main import app  # noqa: E402

if __name__ == "__main__":
    app()
