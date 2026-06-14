"""Provides the 'sle get' subcommand for evaluating the composition of the data acquisition system managed by the
host-machine.
"""

import click
from ataraxis_video_system import CameraInterfaces, discover_camera_ids
from ataraxis_base_utilities import LogLevel, console
from ataraxis_transport_layer_pc import print_available_ports
from ataraxis_communication_interface.interfaces.cli import identify as _identify_microcontrollers

from ..vr_task import UnityBridgeClient
from ..cross_system import CRCCalculator, discover_zaber_devices

CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}  # pragma: no cover
"""Ensures that displayed Click help messages are formatted according to the lab standard."""

_MICROCONTROLLER_BAUDRATE: int = 115200
"""The baud rate used to communicate with the data acquisition system's microcontrollers during discovery."""


@click.group("get", context_settings=CONTEXT_SETTINGS)
def get() -> None:  # pragma: no cover
    """Evaluates the composition of the data acquisition system managed by the host-machine."""


@get.command("zaber")
def get_zaber_devices() -> None:
    """Identifies the Zaber devices accessible to the data acquisition system."""
    discover_zaber_devices()


@get.command("cameras")
def get_cameras() -> None:
    """Identifies the cameras accessible to the data acquisition system."""
    all_cameras = discover_camera_ids()

    # Separates cameras by interface for display purposes.
    opencv_cameras = [camera for camera in all_cameras if camera.interface == CameraInterfaces.OPENCV]
    harvesters_cameras = [camera for camera in all_cameras if camera.interface == CameraInterfaces.HARVESTERS]

    # Displays OpenCV camera information.
    if not opencv_cameras:
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
        console.echo(message="Available OpenCV cameras:", level=LogLevel.SUCCESS)
        for number, camera_data in enumerate(opencv_cameras, start=1):
            console.echo(
                message=(
                    f"OpenCV camera {number}: index={camera_data.camera_index}, "
                    f"frame_height={camera_data.frame_height} pixels, frame_width={camera_data.frame_width} pixels, "
                    f"frame_rate={camera_data.acquisition_frame_rate} frames / second."
                )
            )

    # Displays Harvesters camera information.
    if not harvesters_cameras:
        console.echo(message="No Harvesters-compatible cameras discovered.", level=LogLevel.WARNING)
    else:
        # The Harvesters interface exposes the camera model and serial number, which makes it easy to map discovered
        # indices to physical hardware.
        console.echo(message="Available Harvesters cameras:", level=LogLevel.SUCCESS)
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
    _identify_microcontrollers.callback(baudrate=_MICROCONTROLLER_BAUDRATE)  # type: ignore[misc]


@get.command("ports")
def get_ports() -> None:
    """Identifies the serial communication ports accessible to the data acquisition system."""
    print_available_ports()


@get.command("unity")
def get_unity_bridge() -> None:
    """Checks whether the Unity Editor MCP Bridge is reachable for Virtual Reality task sessions."""
    client = UnityBridgeClient()
    try:
        if client.is_reachable():
            console.echo(message=client.describe_status(), level=LogLevel.SUCCESS)
        else:
            message = (
                "Unity bridge: unreachable. Open the Unity project in the editor to enable Virtual Reality task "
                "control; its MCP bridge starts automatically with the editor."
            )
            console.echo(message=message, level=LogLevel.WARNING)
    finally:
        client.close()


@get.command("checksum")
@click.option(
    "-i",
    "--input-string",
    prompt="Enter the string for which to compute the checksum: ",
    help="The string for which to compute the checksum.",
)
def calculate_crc(input_string: str) -> None:
    """Calculates the CRC32-XFER checksum for the input string."""
    calculator = CRCCalculator()
    crc_checksum = calculator.string_checksum(string=input_string)
    console.echo(message=f"The CRC32-XFER checksum for the input string '{input_string}' is: {crc_checksum}.")
