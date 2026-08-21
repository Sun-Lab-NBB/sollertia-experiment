"""Provides the helper that isolates a single step of a multi-asset shutdown sequence so that one failing step does not
skip the remaining ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console

if TYPE_CHECKING:
    from collections.abc import Callable


def run_shutdown_step(description: str, step: Callable[[], None]) -> None:
    """Executes a single shutdown callable, isolating it so that an error or interrupt does not propagate.

    Acquisition system shutdown sequences tear down several subprocess-backed assets in turn. Allowing an exception or
    a repeated KeyboardInterrupt from one asset to propagate would skip the remaining teardown steps. This would leave
    the orphaned subprocesses to be collected by the garbage collector, which tears down their shared-memory managers
    out of order and cascades into multiprocessing errors. This helper contains each failure so the remaining steps
    still run, while the originally propagating exception (if any) resumes once the shutdown sequence completes.

    Args:
        description: A short gerund phrase naming the step, used to contextualize an error encountered while running it.
        step: The zero-argument callable that performs the shutdown step.
    """
    try:
        step()
    except (Exception, KeyboardInterrupt) as error:
        message = (
            f"Encountered an error while {description} during the shutdown sequence: {error!r}. Continuing with the "
            f"remaining shutdown steps."
        )
        console.echo(message=message, level=LogLevel.ERROR)
