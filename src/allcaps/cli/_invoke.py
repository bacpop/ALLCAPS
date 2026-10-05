"""Calling the pipeline modules in-process.

Every pipeline stage is an ``argparse`` module with a ``main(args)``. The CLI builds the
Namespace and calls it directly rather than shelling out to ``python -m allcaps.X``: one
process, one traceback, no dependence on the working directory, and the base model stays
loaded between stages where that matters.

:func:`ns` fills in a module's defaults from its own parser, so a stage never silently
misses an attribute its ``main`` reads.
"""

import argparse
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..logging_config import get_logger

logger = get_logger(__name__)


def ns(build_parser: Optional[Callable] = None, **overrides: Any) -> Namespace:
    """A Namespace carrying a module's parser defaults plus ``overrides``.

    Defaults are read off the module's own parser rather than re-declared here, so
    adding an argument to a stage cannot leave the CLI handing it an incomplete
    Namespace — which previously surfaced as ``AttributeError`` deep inside a stage.

    Note this deliberately does not call ``parse_args([])``: most of these parsers have
    required arguments, so an empty argv would just exit.
    """
    base: dict = {}
    if build_parser is not None:
        parser = build_parser()
        # `_actions` is the only way to enumerate dests; `get_default` is public.
        base = {
            action.dest: parser.get_default(action.dest)
            for action in parser._actions
            if action.dest not in ("help", argparse.SUPPRESS)
        }
    base.update(overrides)
    return Namespace(**base)


def stage(name: str, func: Callable, args: Any = None, skip_if: Sequence[Path] = ()) -> bool:
    """Run one pipeline stage, with timing and optional resume.

    Returns True when the stage ran, False when it was skipped because its outputs
    already existed.
    """
    outputs = [Path(p) for p in skip_if]
    if outputs and all(p.exists() for p in outputs):
        logger.info("[%s] up to date, skipping (%s)", name, ", ".join(str(p) for p in outputs))
        return False

    logger.info("[%s] starting", name)
    started = time.time()
    if args is None:
        func()
    else:
        func(args)
    logger.info("[%s] done in %.1fs", name, time.time() - started)
    return True


def resolve_device(requested: str) -> str:
    """Turn ``auto`` into a concrete torch device string.

    Only ``cuda`` and ``cpu`` are offered: the model was trained and validated on those
    two, and silently routing to a third backend is not a kindness.
    """
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"
