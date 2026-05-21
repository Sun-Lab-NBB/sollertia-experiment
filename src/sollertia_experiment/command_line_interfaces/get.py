"""Provides the 'sle get' subcommand for evaluating the composition of the data acquisition system managed by the
host-machine.
"""

import click
from natsort_rs import natsort as natsorted  # type: ignore[import-untyped]
from ataraxis_video_system import CameraInterfaces, discover_camera_ids
from ataraxis_base_utilities import LogLevel, console
from ataraxis_transport_layer_pc import print_available_ports
from ataraxis_communication_interface.cli import identify as _identify_microcontrollers

from .mcp_servers import run_get_server
from ..mesoscope_vr import (
    CRCCalculator,
    discover_zaber_devices,
    get_system_configuration_data,
)
from ..shared_components import get_project_experiments

# Ensures that displayed CLICK help messages are formatted according to the lab standard.
CONTEXT_SETTINGS = {"max_content_width": 120}  # pragma: no cover


@click.group("get", context_settings=CONTEXT_SETTINGS)
def get() -> None:  # pragma: no cover
    """Evaluates the composition of the data acquisition system managed by the host-machine."""


@get.command("zaber")
def get_zaber_devices() -> None:
    """Identifies the Zaber devices accessible to the data acquisition system."""
    discover_zaber_devices()


@get.command("projects")
def get_projects() -> None:
    """Identifies the projects accessible to the data acquisition system."""
    system_configuration = get_system_configuration_data()
    projects = natsorted(
        [
            directory.name  # Use .name instead of .stem (they're the same for directories)
            for directory in system_configuration.filesystem.root_directory.iterdir()
            if directory.is_dir() and not directory.name.startswith(".")
        ]
    )
    if projects:
        console.echo(
            f"The {system_configuration.name} data acquisition system is currently configured to acquire data for the "
            f"following projects: {', '.join(projects)}."
        )
    else:
        console.echo(
            f"The {system_configuration.name} data acquisition system is currently not configured to acquire data for "
            f"any projects. To configure the system to support acquiring data for a new project, use the "
            f"'sle configure project' subcommand."
        )


@get.command("experiments")
@click.option(
    "-p",
    "--project",
    type=str,
    required=True,
    help="The name of the project for which to discover the available experiment configurations.",
)
def get_experiments(project: str) -> None:
    """Identifies the target project's experiment configurations accessible to the data acquisition system."""
    system_configuration = get_system_configuration_data()
    experiments = get_project_experiments(project=project, filesystem_configuration=system_configuration.filesystem)
    if experiments:
        console.echo(
            f"The {system_configuration.name} data acquisition system is currently configured to execute the following "
            f"experiments for the {project} project: {', '.join(experiments)}."
        )
    else:
        console.echo(
            f"The {system_configuration.name} data acquisition system is currently not configured to execute any "
            f"experiments for the {project} project. To configure the system to support a new experiment "
            f"configuration, use the 'sle configure experiment' subcommand."
        )


@get.command("cameras")
def get_cameras() -> None:
    """Identifies the cameras accessible to the data acquisition system."""
    # Discovers all compatible cameras from both interfaces.
    all_cameras = discover_camera_ids()

    # Separates cameras by interface for display purposes.
    opencv_cameras = [cam for cam in all_cameras if cam.interface == CameraInterfaces.OPENCV]
    harvesters_cameras = [cam for cam in all_cameras if cam.interface == CameraInterfaces.HARVESTERS]

    # Displays OpenCV camera information.
    if len(opencv_cameras) == 0:
        console.echo(message="No OpenCV-compatible cameras discovered.", level=LogLevel.WARNING)
    else:
        console.echo(
            message=(
                "Warning! Currently, it is impossible to resolve camera models or serial numbers through the "
                "OpenCV interface. It is recommended to check each discovered OpenCV camera via the 'axvs run' "
                "CLI command to precisely map the discovered camera indices to specific camera hardware."
            ),
            level=LogLevel.WARNING,
        )
        console.echo("Available OpenCV cameras:", level=LogLevel.SUCCESS)
        for number, camera_data in enumerate(opencv_cameras, start=1):
            console.echo(
                message=(
                    f"OpenCV camera {number}: index={camera_data.camera_index}, "
                    f"frame_height={camera_data.frame_height} pixels, frame_width={camera_data.frame_width} pixels, "
                    f"frame_rate={camera_data.acquisition_frame_rate} frames / second."
                )
            )

    # Displays Harvesters camera information.
    if len(harvesters_cameras) == 0:
        console.echo(message="No Harvesters-compatible cameras discovered.", level=LogLevel.WARNING)
    else:
        # Note, Harvesters interface supports identifying the camera's model and serial number, which makes it easy to
        # map discovered indices to physical hardware.
        console.echo("Available Harvesters cameras:", level=LogLevel.SUCCESS)
        for number, camera_data in enumerate(harvesters_cameras, start=1):
            console.echo(
                message=(
                    f"Harvesters camera {number}: index={camera_data.camera_index}, model={camera_data.model}, "
                    f"serial_code={camera_data.serial_number}, frame_height={camera_data.frame_height} pixels, "
                    f"frame_width={camera_data.frame_width} pixels, "
                    f"frame_rate={camera_data.acquisition_frame_rate} frames / second."
                )
            )


@get.command("controllers")
def get_microcontrollers() -> None:
    """Identifies the microcontrollers accessible to the data acquisition system."""
    _identify_microcontrollers.callback(baudrate=115200)  # type: ignore[misc]


@get.command("ports")
def get_ports() -> None:
    """Identifies the serial communication ports accessible to the data acquisition system."""
    print_available_ports()


@get.command("checksum")
@click.option(
    "-i",
    "--input_string",
    prompt="Enter the string for which to compute the checksum: ",
    help="The string for which to compute the checksum.",
)
def calculate_crc(input_string: str) -> None:
    """Calculates the CRC32-XFER checksum for the input string."""
    calculator = CRCCalculator()
    crc_checksum = calculator.string_checksum(input_string)
    click.echo(f"The CRC32-XFER checksum for the input string '{input_string}' is: {crc_checksum}.")


@get.command("mcp")
@click.option(
    "-t",
    "--transport",
    type=str,
    default="stdio",
    show_default=True,
    help="The MCP transport type ('stdio', 'sse', or 'streamable-http').",
)
def start_get_mcp_server(transport: str) -> None:  # pragma: no cover
    """Starts the MCP server for agentic access to 'sle get' tools."""
    run_get_server(transport=transport)  # type: ignore[arg-type]
