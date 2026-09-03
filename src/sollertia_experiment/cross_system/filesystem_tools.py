"""Provides the filesystem inspection helpers shared by multiple data acquisition systems."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def probe_writable(path: Path) -> str | None:
    """Probes write access to a directory by creating and removing a uniquely-named temporary file.

    Args:
        path: The directory whose write access is probed.

    Returns:
        None when the directory is writable, or a human-readable reason describing why it is not.
    """
    probe = path.joinpath(f".sollertia_experiment_probe_{uuid.uuid4().hex[:8]}")
    try:
        probe.touch()
        probe.unlink()
    except OSError as exception:
        return str(exception)
    return None
