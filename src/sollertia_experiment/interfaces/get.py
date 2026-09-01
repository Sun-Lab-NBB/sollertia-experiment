"""Provides the 'sle get' subcommand for evaluating the composition of the data acquisition system managed by the
host-machine.
"""

import click
from ataraxis_video_system import (
    GENICAM_UNAVAILABLE_REASON,
    CameraInterfaces,
    check_cti_file,
    discover_camera_ids,
    genicam_runtime_available,
)
from ataraxis_base_utilities import LogLevel, console
from ataraxis_transport_layer_pc import print_available_ports
from ataraxis_communication_interface import discover_microcontrollers

from ..vr_task import UnityBridgeClient
from ..cross_system import CRCCalculator, discover_zaber_devices

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""The Click help-message width applied to every command in this group."""

_MICROCONTROLLER_BAUDRATE: int = 115200
"""The baud rate used to communicate with the data acquisition system's microcontrollers during discovery."""


@click.group("get", context_settings=_CONTEXT_SETTINGS)
def get() -> None:
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

    if not harvesters_cameras:
        # An empty Harvesters listing has four causes, so each branch below names the one that applies, instead of
        # sending the operator to inspect camera cabling and power in every case. The runtime is evaluated first,
        # because check_cti_file() also returns None where the runtime is absent. The check_cti_file() branch covers two
        # of the four causes, an unconfigured .cti path and a configured path that no longer loads.
        if not genicam_runtime_available():
            console.echo(
                message=f"Harvesters camera discovery skipped. {GENICAM_UNAVAILABLE_REASON}", level=LogLevel.WARNING
            )
        elif check_cti_file() is None:
            console.echo(
                message=(
                    "Harvesters camera discovery skipped. No GenTL Producer interface (.cti) file is configured. Use "
                    "the 'axvs cti set' CLI command or the 'AXVS_CTI_PATH' environment variable to configure the file "
                    "before discovering GenICam cameras."
                ),
                level=LogLevel.WARNING,
            )
        else:
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
    # Announces the scan before it runs, because probing every port takes seconds and the operator would otherwise
    # face a silent terminal for the whole scan.
    console.echo(
        message=f"Evaluating serial ports at baudrate {_MICROCONTROLLER_BAUDRATE}, this may take a moment...",
        level=LogLevel.INFO,
    )
    controllers = discover_microcontrollers(baudrate=_MICROCONTROLLER_BAUDRATE)

    if not controllers:
        console.echo(message="No valid serial ports detected.", level=LogLevel.WARNING)
        return

    for number, controller in enumerate(controllers, start=1):
        if controller.error_message is not None:
            status = f"Connection Failed: {controller.error_message}"
        elif controller.controller_id is None:
            status = "No microcontroller"
        else:
            status = f"Microcontroller ID: {controller.controller_id}"
        console.echo(message=f"{number}: {controller.port} -> {controller.description} [{status}]")


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
