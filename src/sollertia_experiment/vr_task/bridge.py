"""Provides the UnityBridgeClient used to drive Unity Editor scene activation and play-mode control from the Virtual
Reality task driver via the editor-only MCP Bridge HTTP endpoint exposed by sollertia-unity-tasks.
"""

from __future__ import annotations

import json
from typing import Literal
import logging
from pathlib import Path

import httpx

# Silences httpx's per-request INFO logging so the bridge's frequent localhost HTTP calls do not flood the runtime
# console with redundant request records.
logging.getLogger("httpx").setLevel(logging.WARNING)

_BRIDGE_HOST: str = "127.0.0.1"
"""The loopback host the sollertia-unity-tasks editor MCP Bridge binds its HTTP listener to. The bridge only accepts
loopback connections, so the runtime must execute on the same machine as the Unity Editor."""

_BRIDGE_PORT: int = 8090
"""The TCP port the sollertia-unity-tasks editor MCP Bridge binds its HTTP listener to."""

_BRIDGE_REQUEST_TIMEOUT_S: float = 5.0
"""The per-request timeout, in seconds, applied to every bridge HTTP call. Kept short because the bridge runs on the
local loopback interface and the runtime must not stall on an unresponsive editor."""


class UnityBridgeError(RuntimeError):
    """Raised when the Unity Editor MCP Bridge is unreachable or returns a failed tool response."""


class UnityBridgeClient:
    """Drives Unity Editor scene activation and play-mode transitions over the editor-only MCP Bridge.

    Wraps the bridge's HTTP JSON tool protocol exposed by McpBridge.cs in sollertia-unity-tasks. Every call POSTs
    a single short-timeout localhost request and raises UnityBridgeError on a connection failure or a failed tool
    response, letting the caller decide between retrying, surfacing the error, or probing reachability.

    Notes:
        Methods raise UnityBridgeError directly rather than through console.error because reachability probing
        catches the error as part of normal control flow. Logging every probe failure as an error would be
        misleading when the editor is simply closed, so error reporting is deferred to the driver layer.

    Args:
        host: The loopback host the bridge listens on.
        port: The TCP port the bridge listens on.

    Attributes:
        _url: The fully-formed bridge endpoint every tool call is POSTed to.
        _client: The httpx client that issues bridge requests.
    """

    def __init__(self, host: str = _BRIDGE_HOST, port: int = _BRIDGE_PORT) -> None:
        self._url: str = f"http://{host}:{port}/"
        self._client: httpx.Client = httpx.Client(timeout=_BRIDGE_REQUEST_TIMEOUT_S)

    def __repr__(self) -> str:
        """Returns a string representation of the UnityBridgeClient instance."""
        return f"UnityBridgeClient(url={self._url})"

    def list_scenes(self) -> tuple[tuple[str, ...], str]:
        """Requests the project-relative paths of all Unity scenes and the active scene.

        Returns:
            A two-element tuple whose first element is the tuple of every scene path in the project and whose
            second element is the active scene's path.

        Raises:
            UnityBridgeError: If the bridge is unreachable or the response omits the expected fields.
        """
        payload = self._call(tool="list_scenes")
        scenes = payload.get("scenes")
        active_scene = payload.get("active_scene")
        if not isinstance(scenes, list) or not isinstance(active_scene, str):
            message = (
                f"Unable to parse the Unity bridge 'list_scenes' response. Expected a 'scenes' list and an "
                f"'active_scene' string, but got {payload}."
            )
            raise UnityBridgeError(message)
        return tuple(str(scene) for scene in scenes), active_scene

    def open_scene(self, scene_path: str, unsaved_changes: Literal["", "save", "discard"] = "save") -> None:
        """Opens the scene at the given project-relative path, applying the unsaved-changes policy.

        Args:
            scene_path: The project-relative path of the scene to open.
            unsaved_changes: The policy applied when the active scene has unsaved edits. A "save" value persists
                them before switching, "discard" abandons them, and an empty value leaves the policy unspecified.

        Raises:
            UnityBridgeError: If the bridge is unreachable or refuses to open the scene.
        """
        self._call(tool="open_scene", args={"scene_path": scene_path, "unsaved_changes": unsaved_changes})

    def enter_play_mode(self) -> str:
        """Requests the editor to enter Play Mode.

        Returns:
            The post-request play state reported by the bridge, either "playing" when the editor was already
            playing or "entering_play_mode" when the transition was just triggered.

        Raises:
            UnityBridgeError: If the bridge is unreachable or the response omits the play state.
        """
        return self._require_state(tool="enter_play_mode")

    def exit_play_mode(self) -> str:
        """Requests the editor to exit Play Mode.

        Returns:
            The post-request play state reported by the bridge, either "edit" when the editor was not playing or
            "exiting_play_mode" when the transition was just triggered.

        Raises:
            UnityBridgeError: If the bridge is unreachable or the response omits the play state.
        """
        return self._require_state(tool="exit_play_mode")

    def get_play_state(self) -> tuple[str, str]:
        """Requests the editor's current play state and active scene name.

        Returns:
            A two-element tuple whose first element is the play state ("playing", "compiling", or "edit") and
            whose second element is the active scene's name without its file extension.

        Raises:
            UnityBridgeError: If the bridge is unreachable or the response omits the expected fields.
        """
        payload = self._call(tool="get_play_state")
        state = payload.get("state")
        active_scene = payload.get("active_scene")
        if not isinstance(state, str) or not isinstance(active_scene, str):
            message = (
                f"Unable to parse the Unity bridge 'get_play_state' response. Expected a 'state' string and an "
                f"'active_scene' string, but got {payload}."
            )
            raise UnityBridgeError(message)
        return state, active_scene

    def resolve_scene_path(self, scene_name: str) -> str:
        """Resolves a scene name to its project-relative path using the project's scene listing.

        Args:
            scene_name: The scene name to resolve, matched against each scene path's stem.

        Returns:
            The project-relative path of the scene whose file stem equals the given name.

        Raises:
            UnityBridgeError: If the bridge is unreachable or no scene with a matching name exists in the project.
        """
        scene_paths, _ = self.list_scenes()
        for scene_path in scene_paths:
            if Path(scene_path).stem == scene_name:
                return scene_path

        available = ", ".join(sorted(Path(scene_path).stem for scene_path in scene_paths))
        message = (
            f"Unable to resolve the Unity scene '{scene_name}' to a project scene path. No scene with a matching "
            f"name exists in the Unity project. Available scenes: {available}."
        )
        raise UnityBridgeError(message)

    def is_reachable(self) -> bool:
        """Returns True when the bridge responds to a play-state probe and False when it raises UnityBridgeError."""
        try:
            self.get_play_state()
        except UnityBridgeError:
            return False
        return True

    def describe_status(self) -> str:
        """Returns a one-line human-readable summary of the bridge's reachability, active scene, and play state."""
        try:
            state, active_scene = self.get_play_state()
        except UnityBridgeError:
            return f"Unity bridge: unreachable at {self._url}"
        return f"Unity bridge: reachable | scene={active_scene} | state={state}"

    def close(self) -> None:
        """Closes the underlying httpx client and releases its connection pool."""
        self._client.close()

    def _require_state(self, tool: str) -> str:
        """Issues a play-mode transition tool call and returns the play state it reports.

        Args:
            tool: The bridge tool name to call, either "enter_play_mode" or "exit_play_mode".

        Returns:
            The play state reported in the bridge response.

        Raises:
            UnityBridgeError: If the bridge is unreachable or the response omits the play state.
        """
        payload = self._call(tool=tool)
        state = payload.get("state")
        if not isinstance(state, str):
            message = (
                f"Unable to parse the Unity bridge '{tool}' response. Expected a 'state' string, but got {payload}."
            )
            raise UnityBridgeError(message)
        return state

    def _call(self, tool: str, args: dict[str, object] | None = None) -> dict[str, object]:
        """POSTs a single tool call to the bridge and returns the parsed success payload.

        Args:
            tool: The bridge tool name to invoke.
            args: The tool arguments forwarded to the bridge, or None to send an empty argument object.

        Returns:
            The parsed JSON response object for a successful tool call.

        Raises:
            UnityBridgeError: If the request fails at the transport layer, the response is not valid JSON, or the
                bridge reports the tool call as unsuccessful.
        """
        request_body: dict[str, object] = {"tool": tool, "args": args if args is not None else {}}
        try:
            response = self._client.post(url=self._url, json=request_body)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exception:
            message = (
                f"Unable to reach the Unity Editor MCP Bridge at {self._url} for the '{tool}' tool call: {exception}."
            )
            raise UnityBridgeError(message) from exception
        except json.JSONDecodeError as exception:
            message = (
                f"The Unity Editor MCP Bridge returned a malformed response for the '{tool}' tool call: {exception}."
            )
            raise UnityBridgeError(message) from exception

        if not isinstance(payload, dict):
            message = (
                f"The Unity Editor MCP Bridge returned a non-object response for the '{tool}' tool call: {payload}."
            )
            raise UnityBridgeError(message)

        if not payload.get("success"):
            error_text = payload.get("error", "no error message was provided")
            message = f"The Unity Editor MCP Bridge failed the '{tool}' tool call: {error_text}."
            raise UnityBridgeError(message)

        return payload
