"""Provides utilities for discovering project and experiment assets and querying runtime version data."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from importlib.metadata import metadata as _metadata

from natsort_rs import natsort as natsorted  # type: ignore[import-untyped]
from sollertia_shared_assets import ProjectData

if TYPE_CHECKING:
    from pathlib import Path


def get_version_data() -> tuple[str, str]:
    """Returns the current Python and sollertia-experiment versions.

    Returns:
        The Python version first, then the sollertia-experiment version.
    """
    sollertia_experiment_version: str = _metadata("sollertia-experiment")["version"]
    python_version: str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return python_version, sollertia_experiment_version


def get_project_experiments(project_directory: Path) -> tuple[str, ...]:
    """Discovers the available experiment configuration files for the target project.

    Args:
        project_directory: The path to the project directory for which to discover the experiment configurations.

    Returns:
        A tuple of naturally-sorted experiment configurations available for the target project.
    """
    project: ProjectData = ProjectData(root=project_directory.parent, project_name=project_directory.name)
    return tuple(natsorted([configuration.stem for configuration in project.experiment_configs()]))
