"""Provides miscellaneous assets shared by other library packages."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from importlib.metadata import metadata as _metadata

from natsort_rs import natsort as natsorted  # type: ignore[import-untyped]
from sollertia_shared_assets import iterate_sessions

if TYPE_CHECKING:
    from pathlib import Path


def get_version_data() -> tuple[str, str]:
    """Returns the current Python and sollertia-experiment versions.

    Returns:
        A tuple of two strings. The first string stores the Python version, and the second string stores the
        sollertia-experiment version.
    """
    # Determines the local Python version and the version of the sollertia-experiment library.
    sollertia_experiment_version = _metadata("sollertia-experiment")["version"]
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return python_version, sollertia_experiment_version


def get_animal_project(animal_id: str, root_directory: Path) -> tuple[str, ...]:
    """Scans the input root directory and returns the names of all projects that include the target animal.

    Notes:
        Project membership is derived from discovered session markers, so a project is reported only when it
        contains at least one acquired session for the target animal.

    Args:
        animal_id: The unique identifier of the animal for which to discover the projects that include this animal.
        root_directory: The path to the root directory that stores all project directories managed by the data
            acquisition system.

    Returns:
        A tuple of naturally-sorted project names that include the target animal.
    """
    matching_projects = {
        session.project_name
        for session in iterate_sessions(root_path=root_directory)
        if session.animal_id == animal_id
    }
    return tuple(natsorted(list(matching_projects)))


def get_project_experiments(project_directory: Path) -> tuple[str, ...]:
    """Discovers the available experiment configuration files for the target project.

    Args:
        project_directory: The path to the project directory for which to discover the experiment configurations.

    Returns:
        A tuple of naturally-sorted experiment configurations available for the target project.
    """
    configuration_path = project_directory.joinpath("configuration")
    return tuple(natsorted([configuration.stem for configuration in configuration_path.glob("*.yaml")]))
