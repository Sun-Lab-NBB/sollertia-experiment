"""Provides the runtime configuration dataclass for the Unity Virtual Reality task driver."""

from dataclasses import dataclass


@dataclass(slots=True)
class VRTaskConfiguration:
    """Stores the configuration used to connect to the Unity game engine that runs the Virtual Reality task.

    Notes:
        This configuration only stores the MQTT broker discovery fields used to reach Unity. The geometric VR
        parameters (cue catalog, corridor geometry, cm-per-unity-unit conversion) are resolved at experiment-start
        from the matching TaskTemplate YAML in the sollertia-shared-assets task templates directory.
    """

    ip: str = "127.0.0.1"
    """The IP address of the MQTT broker used to communicate with the Unity game engine."""
    port: int = 1883
    """The port number of the MQTT broker used to communicate with the Unity game engine."""
