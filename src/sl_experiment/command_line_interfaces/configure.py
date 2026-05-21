"""Provides the 'sl-configure' Command Line Interface (CLI) for authoring the Mesoscope-VR data acquisition system
configuration.
"""

from __future__ import annotations

import click
from sollertia_shared_assets import AcquisitionSystems

from ..mesoscope_vr import create_system_configuration_file

CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.group("configure", context_settings=CONTEXT_SETTINGS)
def configure_cli() -> None:  # pragma: no cover
    """Configures the Mesoscope-VR data acquisition system on the local machine."""


@configure_cli.command("system", context_settings=CONTEXT_SETTINGS)
@click.option(
    "-s",
    "--system",
    type=click.Choice(AcquisitionSystems, case_sensitive=False),
    show_default=True,
    default=AcquisitionSystems.MESOSCOPE_VR,
    help="The name (type) of the data acquisition system for which to create the configuration file.",
)
def generate_system_configuration_file(system: AcquisitionSystems) -> None:  # pragma: no cover
    """Creates the Mesoscope-VR data acquisition system configuration file under the working directory."""
    create_system_configuration_file(system=system)
