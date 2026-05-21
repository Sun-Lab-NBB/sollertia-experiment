"""Provides the LoggingHooks Protocol used by the VRTaskDriver to forward log payloads to the acquisition system's
DataLogger.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@runtime_checkable
class LoggingHooks(Protocol):
    """Defines the contract used by VRTaskDriver to forward Virtual Reality task events to the acquisition system's
    log stream.

    Notes:
        VRTaskDriver does not know the acquisition system's log message codes or DataLogger layout. Acquisition
        systems implement this Protocol to attach their own log codes and forward the payloads to their
        DataLogger.input_queue.
    """

    def log_cue_sequence(self, cue_sequence: NDArray[np.uint8]) -> None:
        """Logs the Virtual Reality wall cue sequence received from Unity."""

    def log_reinforcing_guidance_change(self, *, enabled: bool) -> None:
        """Logs the change of the reinforcing trial guidance mode."""

    def log_aversive_guidance_change(self, *, enabled: bool) -> None:
        """Logs the change of the aversive trial guidance mode."""
