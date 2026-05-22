"""Provides the setup-time configuration assets exposed by the Unity Virtual Reality task driver."""

from __future__ import annotations

from dataclasses import dataclass

from ataraxis_base_utilities import console
from sollertia_shared_assets import TaskTemplate, get_task_templates_directory


@dataclass(frozen=True, slots=True)
class VRTaskConfiguration:
    """Stores the configuration used to connect to the Unity game engine that runs the Virtual Reality task.

    Notes:
        This configuration only stores the MQTT broker discovery fields used to reach Unity. The geometric VR
        parameters (cue catalog, corridor geometry, cm-per-unity-unit conversion) are resolved at experiment-start
        from the matching TaskTemplate YAML in the shared VR task templates directory.
    """

    ip: str = "127.0.0.1"
    """The IP address of the MQTT broker used to communicate with the Unity game engine."""
    port: int = 1883
    """The port number of the MQTT broker used to communicate with the Unity game engine."""


def load_vr_task_template(unity_scene_name: str) -> TaskTemplate:
    """Loads the VR TaskTemplate that corresponds to the given Unity scene name.

    Notes:
        Templates are resolved from the directory configured via the sollertia-shared-assets 'slsa configure
        directory' CLI command. The directory typically points to the local sollertia-unity-tasks repository copy. The
        template file name is expected to match the Unity scene name with a '.yaml' suffix.

    Args:
        unity_scene_name: Stem of a YAML template file stored in the configured task templates directory.

    Returns:
        The TaskTemplate parsed from the matching YAML file.

    Raises:
        FileNotFoundError: If the task templates directory does not contain a YAML file whose stem matches the given
            Unity scene name.
    """
    templates_directory = get_task_templates_directory()
    template_path = templates_directory.joinpath(f"{unity_scene_name}.yaml")
    if not template_path.exists():
        available_templates = sorted([candidate.stem for candidate in templates_directory.glob("*.yaml")])
        message = (
            f"Unable to load the Virtual Reality task template for the Unity scene '{unity_scene_name}'. The expected "
            f"template file does not exist at {template_path}. Available templates: {', '.join(available_templates)}."
        )
        console.error(message=message, error=FileNotFoundError)

    return TaskTemplate.from_yaml(file_path=template_path)
