from enum import StrEnum

from ataraxis_time import PrecisionTimer
from ataraxis_communication_interface import MQTTCommunication

from .system import (
    MesoscopePositions as MesoscopePositions,
    MesoscopeAcquisition as MesoscopeAcquisition,
    get_system_configuration as get_system_configuration,
)
from ..vr_task import VRTaskConfiguration as VRTaskConfiguration
from ..cross_system import wait_for_enter as wait_for_enter

_BROKER_POLL_DELAY_MS: int
_ACK_TIMEOUT_MS: int
_PRELOAD_TIMEOUT_MS: int
_REFERENCE_GENERATION_TIMEOUT_MS: int
_RECOVERY_TIMEOUT_MS: int

class _MesoscopeMQTTTopics(StrEnum):
    ALIVE = "MesoscopeAlive"
    PRELOAD = "MesoscopePreload"
    GENERATE_REFERENCE = "MesoscopeGenerateReference"
    BEGIN_ACQUISITION = "MesoscopeBeginAcquisition"
    ABORT = "MesoscopeAbort"
    RECOVER = "MesoscopeRecover"
    QUERY_STATE = "MesoscopeQueryState"
    STATUS = "MesoscopeStatus"
    ERROR = "MesoscopeError"
    STATE = "MesoscopeState"

class _MesoscopeStatusState(StrEnum):
    RECEIVED = "received"
    PRELOADING = "preloading"
    PRELOAD_COMPLETE = "preload_complete"
    GENERATING_ESTIMATOR = "generating_estimator"
    ACQUIRING_ZSTACK = "acquiring_zstack"
    ARMED = "armed"
    GRABBING = "grabbing"
    STOPPED = "stopped"

class MesoscopeDriver:
    _configuration: VRTaskConfiguration
    _acquisition: MesoscopeAcquisition
    _mqtt: MQTTCommunication
    _polling_timer: PrecisionTimer
    def __init__(self, configuration: VRTaskConfiguration, acquisition: MesoscopeAcquisition) -> None: ...
    def __repr__(self) -> str: ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def await_alive(self) -> None: ...
    def is_alive(self, timeout_ms: int = ...) -> bool: ...
    def preload(self, project: str, animal: str) -> None: ...
    def generate_reference(self) -> None: ...
    def begin_acquisition(self) -> None: ...
    def abort(self) -> None: ...
    def recover(self) -> None: ...
    def query_state(self) -> MesoscopePositions: ...
    def _encode_acquisition(self, *, geometry_only: bool) -> bytes: ...
    def _dispatch_command(
        self,
        command: _MesoscopeMQTTTopics,
        payload: bytes | None = None,
        terminal_state: _MesoscopeStatusState | None = None,
        terminal_timeout_ms: int = ...,
    ) -> None: ...
    def _await_status(self, command: _MesoscopeMQTTTopics, state: _MesoscopeStatusState, timeout_ms: int) -> bool: ...
    def _await_state(self, timeout_ms: int) -> bytes | bytearray | None: ...
    @staticmethod
    def _raise_error(payload: bytes | bytearray) -> None: ...
    @staticmethod
    def _echo_progress(state: str, detail: str | None) -> None: ...
    def _clear_buffer(self) -> None: ...

def check_mesoscope_bridge() -> tuple[bool, str]: ...
