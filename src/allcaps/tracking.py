"""Optional experiment tracking, off by default.

Training used to call ``wandb`` directly. That made ``wandb`` a hard dependency for
everyone — including users who only ever want to serotype an assembly — and it coupled
``train_trihead_transformer.main()`` to its own ``__main__`` block: ``wandb.init()`` lived
in the entry point while ``main()`` called ``wandb.log()``, so importing and calling
``main(args)`` from anywhere else raised.

This module is the single seam. Tracking is **disabled unless explicitly started**, every
call is a no-op in that state, and ``wandb`` is imported lazily so it need not be
installed at all. ``ALLCAPS train --wandb`` (or running the module as a script with
``--wandb``) is what turns it on.

Usage::

    from ..tracking import tracking

    tracking.start(args, enabled=args.wandb)   # once, by the entry point
    tracking.log({"loss": 0.1})                # anywhere; no-op when disabled
    tracking.table(columns=[...], data=[...])  # None when disabled
"""

import os
from typing import Any, Optional, Sequence

from .logging_config import get_logger

logger = get_logger(__name__)

WANDB_PROJECT_NAME = "logistic-trihead-augment"


class _Tracking:
    """A wandb facade that is inert until :meth:`start` succeeds."""

    def __init__(self) -> None:
        self._wandb = None  # the module, once imported
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start(
        self,
        config: Any = None,
        enabled: bool = False,
        project: str = WANDB_PROJECT_NAME,
        run_name: Optional[str] = None,
    ) -> bool:
        """Begin a tracked run. Returns whether tracking is actually on.

        When ``enabled`` is False this does nothing at all — no import, no network, no
        ``wandb`` directory. When True but ``wandb`` is not installed, it warns and
        carries on untracked rather than failing the training run; losing metrics is not
        a reason to lose a 19-hour fit.
        """
        if not enabled:
            logger.info("Experiment tracking disabled; metrics will not be uploaded.")
            return False

        try:
            import wandb  # noqa: PLC0415 — deliberately lazy
        except ImportError:
            logger.warning(
                "--wandb was passed but wandb is not installed; continuing untracked. "
                "Install it with: pip install 'allcaps[wandb]'"
            )
            return False

        # Honour WANDB_MODE=offline, which is how the cluster drivers run.
        mode = "offline" if os.environ.get("WANDB_MODE") == "offline" else "online"
        self._wandb = wandb
        wandb.init(project=project, config=config, mode=mode)
        if run_name is None:
            run_id = os.environ.get("SLURM_JOB_ID", os.urandom(4).hex())
            run_name = f"{project}-{run_id}"
        if wandb.run is not None:
            wandb.run.name = run_name
        self._active = True
        logger.info("Experiment tracking started (mode=%s, run=%s).", mode, run_name)
        return True

    def log(self, data: dict, **kwargs: Any) -> None:
        if self._active and self._wandb is not None:
            self._wandb.log(data, **kwargs)

    def config_update(self, data: dict, **kwargs: Any) -> None:
        if self._active and self._wandb is not None:
            self._wandb.config.update(data, **kwargs)

    def table(self, columns: Sequence[str], data: Sequence[Sequence[Any]]) -> Any:
        """A ``wandb.Table``, or None when untracked.

        Callers can build tables unconditionally: when untracked this returns None
        and the surrounding ``log`` call is itself a no-op, so the None goes nowhere.
        """
        if self._active and self._wandb is not None:
            return self._wandb.Table(columns=list(columns), data=list(data))
        return None

    def finish(self) -> None:
        if self._active and self._wandb is not None:
            self._wandb.finish()
            self._active = False


#: Process-wide tracker. Inert until ``start(..., enabled=True)``.
tracking = _Tracking()
