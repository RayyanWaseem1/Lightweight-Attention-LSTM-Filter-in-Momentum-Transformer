"""Compatibility shim for the removed tuned-config module.

The active configuration source is ``Models.config``. Importing from this
module remains supported so older notebooks/scripts do not keep a stale copy of
hyperparameters or obsolete tuning-result claims.
"""

from __future__ import annotations

try:
    from .config import *  # noqa: F401,F403
except ImportError:  # pragma: no cover - supports direct script execution
    from Models.config import *  # type: ignore # noqa: F401,F403
