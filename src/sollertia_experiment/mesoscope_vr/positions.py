"""Provides per-session runtime position dataclasses specific to the Mesoscope-VR data acquisition system.

These classes store motor stage positions reused between data acquisition sessions. They previously lived in
sollertia-shared-assets but have moved here since the Mesoscope-VR acquisition runtime is the only consumer.
"""

from dataclasses import dataclass

from ataraxis_data_structures import YamlConfig


@dataclass
class ZaberPositions(YamlConfig):  # pragma: no cover
    """Stores Zaber motor positions reused between data acquisition sessions that use the Mesoscope-VR system."""

    headbar_z: int = 0
    """The absolute position, in native motor units, of the HeadBar z-axis motor."""
    headbar_pitch: int = 0
    """The absolute position, in native motor units, of the HeadBar pitch-axis motor."""
    headbar_roll: int = 0
    """The absolute position, in native motor units, of the HeadBar roll-axis motor."""
    lickport_z: int = 0
    """The absolute position, in native motor units, of the LickPort z-axis motor."""
    lickport_y: int = 0
    """The absolute position, in native motor units, of the LickPort y-axis motor."""
    lickport_x: int = 0
    """The absolute position, in native motor units, of the LickPort x-axis motor."""
    wheel_x: int = 0
    """The absolute position, in native motor units, of the running wheel platform x-axis motor."""


@dataclass
class MesoscopePositions(YamlConfig):  # pragma: no cover
    """Stores the positions of real and virtual Mesoscope imaging axes reused between experiment sessions that use the
    Mesoscope-VR system.
    """

    mesoscope_x: float = 0.0
    """The Mesoscope objective's X-axis position, in micrometers."""
    mesoscope_y: float = 0.0
    """The Mesoscope objective's Y-axis position, in micrometers."""
    mesoscope_roll: float = 0.0
    """The Mesoscope objective's Roll-axis position, in degrees."""
    mesoscope_z: float = 0.0
    """The Mesoscope objective's Z-axis position, in micrometers."""
    mesoscope_fast_z: float = 0.0
    """The ScanImage's FastZ (virtual Z-axis) position, in micrometers."""
    mesoscope_tip: float = 0.0
    """The ScanImage's Tip position, in degrees."""
    mesoscope_tilt: float = 0.0
    """The ScanImage's Tilt position, in degrees."""
    laser_power_mw: float = 0.0
    """The laser excitation power at the sample, in milliwatts."""
    red_dot_alignment_z: float = 0.0
    """The Mesoscope objective's Z-axis position, in micrometers, used for the red-dot alignment procedure."""
