"""Provides a cross-system registry and shared helper functions for managing data acquisition system
configuration files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from sollertia_shared_assets import CONFIGURATION_DIRECTORY, AcquisitionSystems, get_working_directory
from ataraxis_data_structures import YamlConfig

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class SystemConfiguration(YamlConfig):
    """Base class for Sollertia data acquisition system configurations.

    Each acquisition system defines a concrete subclass (e.g. MesoscopeSystemConfiguration) that composes its
    per-subsystem configuration sections and registers it via register_system_configuration(). Subclassing provides
    a common type for the cross-system configuration registry and helpers below, plus a default save() that subclasses
    override when the on-disk YAML representation must differ from the in-memory representation.
    """

    def save(self, path: Path) -> None:
        """Saves the configuration instance to disk as a .yaml file.

        Subclasses override this method when the YAML representation must differ from the in-memory representation
        (e.g., to convert a calibration tuple to a mapping for a stable on-disk layout).

        Args:
            path: The path to the .yaml file to save the configuration data to.
        """
        self.to_yaml(file_path=path)


_SYSTEM_CONFIGURATION_CLASSES: dict[AcquisitionSystems, type[SystemConfiguration]] = {}
"""Maps each registered acquisition system to its SystemConfiguration subclass. Populated at import time by
register_system_configuration(), called from each acquisition system's configuration module."""


def register_system_configuration(
    system: AcquisitionSystems | str, configuration_class: type[SystemConfiguration]
) -> None:
    """Registers the SystemConfiguration subclass used by the specified data acquisition system.

    Each acquisition system calls this function at import time so that the cross-system configuration helpers below can
    create, resolve, and load that system's configuration file. Registration is the only system-specific wiring the
    file lifecycle requires; everything else is shared.

    Args:
        system: The acquisition system the configuration class belongs to.
        configuration_class: The system's SystemConfiguration subclass.
    """
    _SYSTEM_CONFIGURATION_CLASSES[AcquisitionSystems(str(system))] = configuration_class


def _system_configuration_filename(system: AcquisitionSystems) -> str:
    """Returns the canonical configuration-file name for the specified acquisition system."""
    return f"{system}_system_configuration.yaml"


def create_system_configuration_file(system: AcquisitionSystems | str) -> None:
    """Creates the default .yaml configuration file for the specified data acquisition system and configures the local
    machine (PC) to use it for all future acquisition-system-related calls.

    The file is written into the local working directory's configuration folder. Any pre-existing system configuration
    file is removed first, so the machine always belongs to exactly one acquisition system.

    Args:
        system: The acquisition system to create the configuration file for.

    Raises:
        ValueError: If the requested acquisition system is not registered.
    """
    resolved = AcquisitionSystems(str(system))
    if resolved not in _SYSTEM_CONFIGURATION_CLASSES:
        supported = ", ".join(str(member) for member in _SYSTEM_CONFIGURATION_CLASSES) or "none"
        message = (
            f"Unable to generate the system configuration file for the acquisition system '{system}'. The requested "
            f"acquisition system is not registered. Currently registered acquisition systems: {supported}."
        )
        console.error(message=message, error=ValueError)

    directory = get_working_directory().joinpath(CONFIGURATION_DIRECTORY)
    ensure_directory_exists(path=directory)

    # Removes any existing system configuration file(s) so that exactly one remains after this call.
    for existing in tuple(directory.glob("*_system_configuration.yaml")):
        console.echo(message=f"Removing the existing configuration file {existing.name}...", level=LogLevel.INFO)
        existing.unlink()

    configuration_path = directory.joinpath(_system_configuration_filename(resolved))
    _SYSTEM_CONFIGURATION_CLASSES[resolved]().save(path=configuration_path)

    message = (
        f"{resolved} data acquisition system configuration file: Saved to {configuration_path}. Edit the default "
        f"parameters inside the configuration file to finish configuring the system."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)


def get_system_configuration_path() -> Path:
    """Resolves the path to the local machine's data acquisition system configuration file.

    A host-machine belongs to exactly one acquisition system, so its working directory must contain exactly one
    ``*_system_configuration.yaml`` file.

    Returns:
        The path to the single local system configuration file.

    Raises:
        FileNotFoundError: If the local working directory does not contain exactly one system configuration file.
    """
    directory = get_working_directory().joinpath(CONFIGURATION_DIRECTORY)
    configuration_files = tuple(directory.glob("*_system_configuration.yaml"))

    if len(configuration_files) != 1:
        found = ", ".join(file.name for file in configuration_files) if configuration_files else "none"
        message = (
            f"Unable to resolve the local data acquisition system configuration file. Expected exactly one "
            f"'*_system_configuration.yaml' file inside {directory}, but found {len(configuration_files)} ({found}). "
            f"Use the system's 'configure' CLI command (e.g. 'sle mesoscope configure system') to reconfigure the "
            f"host-machine to belong to exactly one acquisition system."
        )
        console.error(message=message, error=FileNotFoundError)

    return configuration_files[0]


def get_system_configuration_data() -> SystemConfiguration:
    """Loads the local machine's data acquisition system configuration file.

    Resolves the single configuration file on the local machine, maps it to the registered SystemConfiguration subclass
    for its acquisition system, and returns the loaded instance.

    Notes:
        The return type is the shared ``SystemConfiguration`` base. Callers that need a specific system's concrete type
        use that system's typed wrapper (e.g. ``sollertia_experiment.mesoscope_vr.get_system_configuration``), which
        validates the loaded configuration's type and narrows it.

    Returns:
        The loaded SystemConfiguration instance.

    Raises:
        FileNotFoundError: If the local working directory does not contain exactly one system configuration file.
        ValueError: If the configuration file does not belong to a registered acquisition system.
    """
    configuration_path = get_system_configuration_path()

    for system, configuration_class in _SYSTEM_CONFIGURATION_CLASSES.items():
        if configuration_path.name == _system_configuration_filename(system):
            return configuration_class.from_yaml(file_path=configuration_path)

    supported = ", ".join(_system_configuration_filename(member) for member in _SYSTEM_CONFIGURATION_CLASSES) or "none"
    message = (
        f"The local data acquisition system configuration file '{configuration_path.name}' does not belong to any "
        f"registered acquisition system. Registered configuration files: {supported}."
    )
    console.error(message=message, error=ValueError)
    # Unreachable: console.error() is NoReturn, but ruff cannot trace NoReturn through method calls (RET503).
    raise ValueError(message)  # pragma: no cover
