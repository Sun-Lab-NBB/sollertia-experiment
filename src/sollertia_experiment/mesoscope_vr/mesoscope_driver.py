"""Provides the MesoscopeDriver class that encapsulates all MQTT communication with the ScanImage software used to
control the Mesoscope during a data acquisition runtime.
"""

from __future__ import annotations

from enum import StrEnum
import json
from typing import TYPE_CHECKING

from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import LogLevel, console
from ataraxis_communication_interface import MQTTCommunication

from .system import MesoscopePositions

if TYPE_CHECKING:
    from pathlib import Path

    from .system import MesoscopeAcquisition
    from ..vr_task import VRTaskConfiguration


_BROKER_POLL_DELAY_MS: int = 10
"""The delay, in milliseconds, between consecutive ScanImagePC status-buffer polls during the command handshake."""

_ACK_TIMEOUT_MS: int = 5000
"""The maximum time, in milliseconds, to wait for the ScanImagePC to acknowledge a command before resending it. The
estimator and z-stack acquisitions take far longer than this, but a command's reception is acknowledged immediately."""


class _MesoscopeMQTTTopics(StrEnum):
    """Defines the set of MQTT topics used to communicate with the ScanImage software that controls the Mesoscope.

    Notes:
        The catalog occupies a flat PascalCase namespace prefixed with 'Mesoscope' that does not overlap with the
        Unity Virtual Reality task topics, so both surfaces share a single MQTT broker. The acquisition runtime
        publishes the command topics; the ScanImagePC publishes the status and error topics.
    """

    ALIVE = "MesoscopeAlive"
    """Liveness probe published by the VRPC (empty payload). The ScanImagePC replies on the Status topic with a
    reception acknowledgement; the VRPC treats the absence of a reply within the acknowledgement timeout as the
    runAcquisition command loop not running. Mirrors the request-reply presence check used for the Unity bridge."""
    PRELOAD = "MesoscopePreload"
    """Request to preload a persisted reference estimator as an alignment aid, carrying the estimator path or null in a
    'path' field. Automatic motion correction stays disabled so the operator enables it manually during alignment."""
    GENERATE_REFERENCE = "MesoscopeGenerateReference"
    """Request to run the lengthy reference sequence (fresh session estimator plus high-definition z-stack) and arm the
    Mesoscope. Carries the full acquisition parameter set as a JSON payload. Dispatched once the acquisition runtime
    detects the alignment screenshot."""
    BEGIN_ACQUISITION = "MesoscopeBeginAcquisition"
    """Request to begin acquiring session frames (empty payload). The TTL frame stream, not this command, confirms
    that frame acquisition actually started."""
    ABORT = "MesoscopeAbort"
    """Request to abort or end the ongoing frame acquisition (empty payload). The TTL frame stream confirms the stop."""
    RECOVER = "MesoscopeRecover"
    """Request to reload the session estimator from the shared data directory and re-arm the Mesoscope without
    regenerating the z-stack. Carries only the plane-geometry acquisition parameters needed to re-derive the imaging
    planes as a JSON payload. Used to resume an acquisition interrupted by a transient failure."""
    QUERY_STATE = "MesoscopeQueryState"
    """Request for a one-shot snapshot of the Mesoscope stage, fast-Z, and laser state (empty payload). The
    ScanImagePC replies on the State topic; used to populate a MesoscopePositions instance at runtime boundaries."""
    STATUS = "MesoscopeStatus"
    """Acknowledgement and progress reply published by the ScanImagePC, carrying 'command', 'state', and optional
    'detail' fields."""
    ERROR = "MesoscopeError"
    """Failure reply published by the ScanImagePC, carrying a 'message' field describing the error."""
    STATE = "MesoscopeState"
    """State snapshot published by the ScanImagePC in reply to a QueryState request, carrying the 'x', 'y', 'r', 'z',
    'fast_z', 'tip', 'tilt', and 'power_mW' fields. 'tip' and 'tilt' are hardware placeholders reported as zero."""


class _MesoscopeStatusState(StrEnum):
    """Defines the state values reported by the ScanImagePC on the MesoscopeStatus topic."""

    RECEIVED = "received"
    """The ScanImagePC received the command and started processing it. Acknowledges command reception."""
    PRELOADING = "preloading"
    """The ScanImagePC is loading the persisted reference estimator."""
    PRELOAD_COMPLETE = "preload_complete"
    """The ScanImagePC finished loading the persisted estimator with automatic correction left disabled."""
    GENERATING_ESTIMATOR = "generating_estimator"
    """The ScanImagePC is acquiring the reference volume and generating the fresh session estimator."""
    ACQUIRING_ZSTACK = "acquiring_zstack"
    """The ScanImagePC is acquiring the high-definition reference z-stack."""
    ARMED = "armed"
    """The ScanImagePC armed the Mesoscope and is ready to begin frame acquisition."""
    GRABBING = "grabbing"
    """The ScanImagePC started acquiring session frames."""
    STOPPED = "stopped"
    """The ScanImagePC stopped frame acquisition."""


class MesoscopeDriver:
    """Drives the ScanImage software that controls the Mesoscope over the shared Virtual Reality MQTT broker.

    Encapsulates the MQTT command contract with the runAcquisition MATLAB function running on the ScanImagePC:
    connection lifecycle, the estimator-preload and reference-generation setup handshake, and the begin, abort, and
    recover acquisition commands. Mesoscope control is tightly coupled to the Virtual Reality task, so the driver
    reuses the Virtual Reality broker discovery configuration rather than defining its own. Each command is dispatched
    with a reception acknowledgement and, where applicable, a terminal-state confirmation; the actual frame
    acquisition is confirmed by the caller through the hardware TTL frame stream, not over MQTT.

    Args:
        configuration: The Virtual Reality task configuration that defines the shared MQTT broker discovery fields.
        acquisition: The Mesoscope acquisition configuration that defines the motion-estimation and z-stack parameters
            delivered to the ScanImagePC with each command that consumes them.

    Attributes:
        _configuration: The VRTaskConfiguration instance that defines the shared MQTT broker discovery fields.
        _acquisition: The MesoscopeAcquisition instance that defines the parameters delivered to the ScanImagePC with
            the reference-generation and recovery commands.
        _mqtt: The MQTTCommunication instance that bidirectionally transfers data between this driver and the
            ScanImagePC.
        _polling_timer: The PrecisionTimer used to delay between consecutive status-buffer polls during the command
            handshake.
    """

    def __init__(self, configuration: VRTaskConfiguration, acquisition: MesoscopeAcquisition) -> None:
        self._configuration: VRTaskConfiguration = configuration
        self._acquisition: MesoscopeAcquisition = acquisition

        # The ScanImagePC replies on the Status topic for commands (including the liveness probe), on the Error topic
        # for failures, and on the State topic for a state query, so the driver monitors those three reply topics.
        monitored_topics: tuple[_MesoscopeMQTTTopics, ...] = (
            _MesoscopeMQTTTopics.STATUS,
            _MesoscopeMQTTTopics.ERROR,
            _MesoscopeMQTTTopics.STATE,
        )
        self._mqtt: MQTTCommunication = MQTTCommunication(
            ip=configuration.ip,
            port=configuration.port,
            monitored_topics=monitored_topics,
        )
        self._polling_timer: PrecisionTimer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

    def __repr__(self) -> str:
        """Returns a string representation of the MesoscopeDriver instance."""
        return f"MesoscopeDriver(ip={self._configuration.ip}, port={self._configuration.port})"

    def connect(self) -> None:
        """Establishes the MQTT connection to the ScanImagePC."""
        self._mqtt.connect()

    def disconnect(self) -> None:
        """Closes the MQTT connection to the ScanImagePC."""
        self._mqtt.disconnect()

    def await_alive(self) -> None:
        """Blocks until the ScanImagePC's runAcquisition MQTT client replies to a liveness probe.

        Notes:
            The driver probes liveness by publishing a MesoscopeAlive request and waiting for the ScanImagePC to
            acknowledge it on the Status topic, mirroring the request-reply presence check used for the Unity bridge.
            A reply confirms that the command loop is running, while the absence of a reply within the acknowledgement
            timeout indicates that it is not. The probe is resent on each timeout because the operator must launch the
            runAcquisition function manually; it requires the ScanImage handles and interactive imaging-parameter
            confirmations.
        """
        message = (
            "Launch the 'runAcquisition(hSI, hSICtl, <parameters>)' function in the MATLAB command line on the "
            "ScanImagePC to arm the mesoscope control interface."
        )
        console.echo(message=message, level=LogLevel.INFO)
        self._dispatch_command(command=_MesoscopeMQTTTopics.ALIVE)
        console.echo(message="Mesoscope control interface: Connected.", level=LogLevel.SUCCESS)

    def preload(self, estimator_path: Path | None) -> None:
        """Instructs the ScanImagePC to preload the persisted reference estimator as an alignment aid.

        Notes:
            The ScanImagePC enables the motion manager so the estimator is visible but leaves automatic correction
            disabled; the operator enables correction manually while aligning the Mesoscope.

        Args:
            estimator_path: The path to the persisted reference estimator on the shared mesoscope directory, or None
                when no persisted estimator exists for the animal (for example, on the first imaging day).
        """
        path_value = str(estimator_path) if estimator_path is not None else None
        payload = json.dumps(obj={"path": path_value}).encode("utf-8")
        self._dispatch_command(
            command=_MesoscopeMQTTTopics.PRELOAD,
            payload=payload,
            terminal_state=_MesoscopeStatusState.PRELOAD_COMPLETE,
        )
        console.echo(message="Mesoscope reference estimator: Preloaded.", level=LogLevel.SUCCESS)

    def generate_reference(self) -> None:
        """Instructs the ScanImagePC to generate the fresh session estimator and high-definition z-stack and arm.

        Notes:
            This runs the lengthy reference sequence, so the method blocks until the ScanImagePC reports that the
            Mesoscope is armed, surfacing the intermediate progress states to the operator while it waits. The command
            carries the full acquisition parameter set, which the ScanImagePC uses to resolve the imaging geometry and
            acquisition settings.
        """
        self._dispatch_command(
            command=_MesoscopeMQTTTopics.GENERATE_REFERENCE,
            payload=self._encode_acquisition(geometry_only=False),
            terminal_state=_MesoscopeStatusState.ARMED,
        )
        console.echo(message="Mesoscope reference: Generated. Mesoscope armed.", level=LogLevel.SUCCESS)

    def begin_acquisition(self) -> None:
        """Instructs the ScanImagePC to begin acquiring session frames.

        Notes:
            The method returns once the ScanImagePC acknowledges the command. The caller confirms that frame
            acquisition actually started through the hardware TTL frame stream rather than over MQTT.
        """
        self._dispatch_command(command=_MesoscopeMQTTTopics.BEGIN_ACQUISITION)

    def abort(self) -> None:
        """Instructs the ScanImagePC to abort or end the ongoing frame acquisition.

        Notes:
            The method returns once the ScanImagePC acknowledges the command. The caller confirms that frame
            acquisition actually stopped through the hardware TTL frame stream rather than over MQTT.
        """
        self._dispatch_command(command=_MesoscopeMQTTTopics.ABORT)

    def recover(self) -> None:
        """Instructs the ScanImagePC to reload the session estimator and re-arm the Mesoscope after an interruption.

        Notes:
            The ScanImagePC reloads the session estimator from the shared data directory and skips the z-stack
            regeneration, so the method blocks only until the Mesoscope is re-armed. The command carries only the
            plane-geometry parameters, which the ScanImagePC uses to re-derive the imaging planes when re-arming.
        """
        self._dispatch_command(
            command=_MesoscopeMQTTTopics.RECOVER,
            payload=self._encode_acquisition(geometry_only=True),
            terminal_state=_MesoscopeStatusState.ARMED,
        )
        console.echo(message="Mesoscope acquisition: Recovered. Mesoscope re-armed.", level=LogLevel.SUCCESS)

    def query_state(self) -> MesoscopePositions:
        """Queries the current Mesoscope imaging positions from the ScanImagePC.

        Notes:
            The driver publishes a QueryState request and waits for the ScanImagePC to reply on the State topic,
            resending the request on each timeout because MQTT messages are published without delivery guarantees. The
            reply captures the live hardware state, so the mesoscope must be connected and idle in its command loop when
            this method is called. The red-dot alignment Z position cannot be queried, so it is left at its default and
            must be populated by the caller.

        Returns:
            A MesoscopePositions instance populated from the ScanImagePC reply, with red_dot_alignment_z left at its
            default placeholder value.
        """
        while True:
            self._clear_buffer()
            self._mqtt.send_data(topic=_MesoscopeMQTTTopics.QUERY_STATE)
            payload = self._await_state(timeout_ms=_ACK_TIMEOUT_MS)
            if payload is not None:
                state = json.loads(payload.decode("utf-8"))
                return MesoscopePositions(
                    mesoscope_x=float(state["x"]),
                    mesoscope_y=float(state["y"]),
                    mesoscope_roll=float(state["r"]),
                    mesoscope_z=float(state["z"]),
                    mesoscope_fast_z=float(state["fast_z"]),
                    mesoscope_tip=float(state["tip"]),
                    mesoscope_tilt=float(state["tilt"]),
                    laser_power_mw=float(state["power_mW"]),
                )
            message = (
                f"The mesoscope control driver requested a state snapshot from the ScanImagePC but received no reply "
                f"within {_ACK_TIMEOUT_MS // 1000} seconds. Ensure the runAcquisition function is running and idle on "
                f"the ScanImagePC."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            input("Enter anything to retry: ")

    def _encode_acquisition(self, *, geometry_only: bool) -> bytes:
        """Serializes the acquisition parameters a command consumes into a JSON MQTT payload.

        Notes:
            Each command ships only the parameters its ScanImagePC handler consumes. The plane-geometry parameters
            resolve the target and reference imaging planes and are required by both reference generation and recovery,
            while the remaining parameters are only required when generating a fresh reference.

        Args:
            geometry_only: Determines whether to include only the plane-geometry parameters (True) or the full
                acquisition parameter set (False).

        Returns:
            The UTF-8 encoded JSON payload carrying the requested acquisition parameters.
        """
        configuration = self._acquisition
        payload: dict[str, object] = {
            "z_step_um": configuration.z_step_um,
            "z_range_um": list(configuration.z_range_um),
            "z_exclusion_um": list(configuration.z_exclusion_um),
            "acquisition_order": configuration.acquisition_order.value,
        }
        if not geometry_only:
            payload["registration_channel"] = configuration.registration_channel
            payload["field_curvature_correction"] = configuration.field_curvature_correction
            payload["frames_per_reference_plane"] = configuration.frames_per_reference_plane
            payload["zstack_scale_factor"] = configuration.zstack_scale_factor
        return json.dumps(obj=payload).encode("utf-8")

    def _dispatch_command(
        self,
        command: _MesoscopeMQTTTopics,
        payload: bytes | None = None,
        terminal_state: _MesoscopeStatusState | None = None,
    ) -> None:
        """Publishes a command to the ScanImagePC and resolves its acknowledgement and optional terminal state.

        Notes:
            The command is resent on each acknowledgement timeout because MQTT messages are published without delivery
            guarantees. Once the ScanImagePC acknowledges reception, the method optionally blocks for the terminal
            state, surfacing the intermediate progress states to the operator.

        Args:
            command: The command topic to publish to the ScanImagePC.
            payload: The encoded command payload, or None to send an empty command.
            terminal_state: The status state that marks the command complete, or None when only the reception
                acknowledgement is required.
        """
        while True:
            self._clear_buffer()
            self._mqtt.send_data(topic=command, payload=payload)
            if self._await_status(command=command, state=_MesoscopeStatusState.RECEIVED, timeout_ms=_ACK_TIMEOUT_MS):
                break
            message = (
                f"The mesoscope control driver sent the '{command}' command to the ScanImagePC but received no "
                f"acknowledgement within {_ACK_TIMEOUT_MS // 1000} seconds. Ensure the runAcquisition function is "
                f"running on the ScanImagePC."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            input("Enter anything to retry: ")

        if terminal_state is not None:
            self._await_status(command=command, state=terminal_state, timeout_ms=None)

    def _await_status(
        self, command: _MesoscopeMQTTTopics, state: _MesoscopeStatusState, timeout_ms: int | None
    ) -> bool:
        """Polls the ScanImagePC status buffer for a matching status message, optionally bounded by a timeout.

        Notes:
            Status messages that report a different state for the same command are surfaced to the operator as
            progress updates. A MesoscopeError message raises a RuntimeError through the console.

        Args:
            command: The command topic whose status messages are awaited.
            state: The status state that resolves the wait.
            timeout_ms: The maximum time, in milliseconds, to wait, or None to wait indefinitely.

        Returns:
            True if a status message with the awaited state arrived in time, False if the timeout elapsed first.
        """
        self._polling_timer.reset()
        while timeout_ms is None or self._polling_timer.elapsed < timeout_ms:
            self._polling_timer.delay(delay=_BROKER_POLL_DELAY_MS, block=False)
            data = self._mqtt.get_data()
            if data is None:
                continue

            topic, payload = data
            if topic == _MesoscopeMQTTTopics.ERROR:
                self._raise_error(payload=payload)
            if topic != _MesoscopeMQTTTopics.STATUS:
                continue

            status = json.loads(payload.decode("utf-8"))
            if status.get("command") != command:
                continue
            if status.get("state") == state:
                return True

            # Surfaces the intermediate progress states (for example, estimator generation and z-stack acquisition)
            # to the operator while waiting for the terminal state.
            self._echo_progress(state=status.get("state", ""), detail=status.get("detail"))
        return False

    def _await_state(self, timeout_ms: int) -> bytes | bytearray | None:
        """Polls the ScanImagePC reply buffer for a state snapshot, bounded by a timeout.

        Notes:
            A MesoscopeError reply raises a RuntimeError through the console.

        Args:
            timeout_ms: The maximum time, in milliseconds, to wait for the state reply.

        Returns:
            The State topic payload if it arrived in time, or None if the timeout elapsed first.
        """
        self._polling_timer.reset()
        while self._polling_timer.elapsed < timeout_ms:
            self._polling_timer.delay(delay=_BROKER_POLL_DELAY_MS, block=False)
            data = self._mqtt.get_data()
            if data is None:
                continue

            topic, payload = data
            if topic == _MesoscopeMQTTTopics.ERROR:
                self._raise_error(payload=payload)
            if topic == _MesoscopeMQTTTopics.STATE:
                return payload
        return None

    @staticmethod
    def _raise_error(payload: bytes | bytearray) -> None:
        """Raises a RuntimeError describing a failure reported by the ScanImagePC on the MesoscopeError topic."""
        error = json.loads(payload.decode("utf-8"))
        message = (
            f"The ScanImagePC reported a mesoscope control error: {error.get('message', 'no detail was provided')}."
        )
        console.error(message=message, error=RuntimeError)

    @staticmethod
    def _echo_progress(state: str, detail: str | None) -> None:
        """Surfaces an intermediate ScanImagePC status state to the operator as a progress message."""
        message = f"Mesoscope status: {state}.{f' {detail}' if detail else ''}"
        console.echo(message=message, level=LogLevel.INFO)

    def _clear_buffer(self) -> None:
        """Drains all pending messages from the MQTT buffer used to communicate with the ScanImagePC."""
        while self._mqtt.has_data:
            _ = self._mqtt.get_data()
