"""Provides miscellaneous assets shared by other library packages."""

import sys

from natsort_rs import natsort as natsorted  # type: ignore[import-untyped]
from importlib_metadata import metadata as _metadata

from ..mesoscope_vr.configuration import MesoscopeFileSystem, get_system_configuration_data


def get_version_data() -> tuple[str, str]:
    """Returns the current Python and sollertia-experiment versions.

    Returns:
        A tuple of two strings. The first string stores the Python version, and the second string stores the
        sollertia-experiment version.
    """
    # Determines the local Python version and the version of the sollertia-experiment library.
    sollertia_experiment_version = _metadata("sollertia-experiment")["version"]  # type: ignore[index]
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return python_version, sollertia_experiment_version


def get_animal_project(animal_id: str) -> tuple[str, ...]:
    """Scans the local project directory and returns the names of all projects that use the target animal.

    Args:
        animal_id: The unique identifier of the animal for which to discover the projects that use this animal.

    Returns:
        A tuple of project names that use the target animal.
    """
    system_configuration = get_system_configuration_data()

    return tuple(
        directory.name
        for directory in system_configuration.filesystem.root_directory.iterdir()
        if directory.is_dir() and directory.joinpath(animal_id).exists()
    )


def get_project_experiments(project: str, filesystem_configuration: MesoscopeFileSystem) -> tuple[str, ...]:
    """Discovers the available experiment configuration files for the target project.

    Args:
        project: The name of the project for which to discover the experiment configurations.
        filesystem_configuration: The MesoscopeFileSystem instance that stores the filesystem configuration of the
            data acquisition system used to acquire the specified project's data.

    Returns:
        A tuple of naturally-sorted experiment configurations available for the target project.
    """
    configuration_path = filesystem_configuration.root_directory.joinpath(project, "configuration")
    return tuple(natsorted([configuration.stem for configuration in configuration_path.glob("*.yaml")]))
