"""Provides MCP tools wrapping the Mesoscope-VR-specific 'sle mesoscope' configuration and data-management logic."""

from __future__ import annotations

import os
from typing import Any, Literal
from pathlib import Path

import yaml
from ataraxis_video_system import GenicamConfiguration, read_camera_configuration
from sollertia_shared_assets import (
    RAW_DATA_DIRECTORY,
    CONFIGURATION_DIRECTORY,
    SessionData,
    RawDataFiles,
    AcquisitionSystems,
    MesoscopeRawDataFiles,
    get_data_root,
    get_working_directory,
)

from .mcp_instance import mcp, read_yaml, serialize, probe_writable, describe_dataclass, write_yaml_validated
from ..mesoscope_vr import (
    EYE_TRACKING_PROJECT_NAME,
    ZaberPositions,
    MesoscopePositions,
    MesoscopeSystemConfiguration,
    purge_session,
    check_mesoscope_bridge,
    preprocess_session_data,
    get_system_configuration,
    get_system_configuration_path,
    migrate_animal_between_projects,
)
# The eye-tracking token is imported from the module that enforces it during preprocessing, so the validation
# reported here cannot drift away from the requirement the preprocessing pipeline actually applies.

_SYSTEM_CONFIGURATION_GLOB: str = "*_system_configuration.yaml"
"""Glob pattern that matches the system configuration file of any acquisition system under the working directory's
configuration directory."""

_MESOSCOPE_SYSTEM_CONFIGURATION_FILENAME: str = f"{AcquisitionSystems.MESOSCOPE_VR}_system_configuration.yaml"
"""Canonical filename of the Mesoscope-VR system configuration file under the working directory's configuration
directory."""


@mcp.tool()
def read_system_configuration_tool() -> dict[str, Any]:
    """Loads the Mesoscope-VR system configuration YAML from the working directory.

    Returns:
        A dictionary with ``data`` (the serialized MesoscopeSystemConfiguration payload) and ``file_path``, or
        ``{"error": ...}`` on failure.
    """
    try:
        instance = get_system_configuration()
    except Exception as exception:
        return {"error": str(exception)}
    return {"file_path": str(get_system_configuration_path()), "data": serialize(value=instance)}


@mcp.tool()
def write_system_configuration_tool(
    configuration_payload: dict[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Creates or replaces the Mesoscope-VR system configuration YAML in the working directory.

    Args:
        configuration_payload: The complete MesoscopeSystemConfiguration payload.
        overwrite: Determines whether to overwrite an existing system configuration file.

    Returns:
        A dictionary with ``file_path`` and ``data`` (the validated configuration payload), or ``{"error": ...}``
        on failure.
    """
    try:
        directory = get_working_directory().joinpath(CONFIGURATION_DIRECTORY)
        # A host-machine that has not been bound to an acquisition system yet holds no configuration file for the shared
        # resolver to find, so the destination is built the way the 'sle mesoscope configure system' command builds it.
        # A directory holding several configuration files goes through the resolver instead, which rejects the ambiguity
        # and names the files it found.
        if not tuple(directory.glob(_SYSTEM_CONFIGURATION_GLOB)):
            file_path = directory.joinpath(_MESOSCOPE_SYSTEM_CONFIGURATION_FILENAME)
        else:
            file_path = get_system_configuration_path()
    except Exception as exception:
        return {"error": str(exception)}
    return write_yaml_validated(
        file_path=file_path,
        payload=configuration_payload,
        validator_cls=MesoscopeSystemConfiguration,
        overwrite=overwrite,
        use_save_method=True,
    )


@mcp.tool()
def validate_system_configuration_tool() -> dict[str, Any]:
    """Validates the active Mesoscope-VR system configuration and reports filesystem mount status.

    Notes:
        Beyond the per-path mount checks, this verifies that a configured DeepLabCut project is able to produce the
        eye-tracking predictions expected by session preprocessing. Every finding is reported through the ``issues``
        list, and none of them alters any acquisition behavior.

    Returns:
        A dictionary with ``valid``, ``issues``, and ``paths`` (the per-path mount report), or ``{"error": ...}`` on
        failure.
    """
    try:
        configuration = get_system_configuration()
    except Exception as exception:
        return {"error": str(exception)}

    paths = _filesystem_paths_report(configuration=configuration)
    issues = [
        f"{name}: {report.get('error', 'not ok')}" for name, report in paths.items() if not report.get("ok", False)
    ]

    # The face-camera inference contract reaches past the project file resolving, so a project that resolves is
    # inspected further. An unset or unreadable project is already covered by the path report above.
    if configuration.video_tracking.dlc_project_path != Path() and paths["dlc_project"].get("ok", False):
        task_issue = _check_dlc_project_task(project_path=configuration.video_tracking.dlc_project_path)
        if task_issue is not None:
            issues.append(f"dlc_project: {task_issue}")

    return {"valid": not issues, "issues": issues, "paths": paths}


@mcp.tool()
def verify_camera_configuration_tool() -> dict[str, Any]:
    """Compares each camera's live GenICam configuration against its stored configuration .yaml file.

    For every camera in the active Mesoscope-VR system configuration that declares a configuration file path, this
    tool connects to the camera, dumps its current GenICam node configuration, and diffs it against the stored
    configuration file. Cameras whose configuration path is unset are reported as not configured.

    Returns:
        A dictionary with a ``cameras`` key mapping each camera role to its verification report, or ``{"error": ...}``
        if the active system configuration cannot be loaded.
    """
    try:
        configuration = get_system_configuration()
    except Exception as exception:
        return {"error": str(exception)}

    cameras = configuration.cameras
    targets = (
        ("face_camera", cameras.face_camera_index, cameras.face_camera_configuration_path),
        ("body_camera", cameras.body_camera_index, cameras.body_camera_configuration_path),
    )
    report: dict[str, Any] = {
        role: _verify_single_camera(camera_index=camera_index, configuration_path=configuration_path)
        for role, camera_index, configuration_path in targets
    }
    return {"cameras": report}


@mcp.tool()
def describe_system_configuration_schema_tool() -> dict[str, Any]:
    """Returns the schema for MesoscopeSystemConfiguration and its nested hardware dataclasses.

    Returns:
        A dictionary with a single ``schema`` key holding the recursive field description of
        MesoscopeSystemConfiguration and every nested hardware dataclass.
    """
    return {"schema": describe_dataclass(cls=MesoscopeSystemConfiguration)}


@mcp.tool()
def check_system_mounts_tool() -> dict[str, Any]:
    """Verifies all filesystem paths declared in the active Mesoscope-VR system configuration.

    Notes:
        The report covers the platform data root, the mesoscope acquisition directory, every configured long-term
        storage destination, the two stored camera GenICam configurations, and the DeepLabCut project that drives
        face-camera inference. Paths belonging to a feature the host leaves switched off report as not configured.

    Returns:
        A dictionary with ``system_name``, ``paths`` (the per-path diagnostic report), and ``summary`` (the count
        of reachable and failed paths), or ``{"error": ...}`` on failure.
    """
    try:
        configuration = get_system_configuration()
    except Exception as exception:
        return {"error": str(exception)}

    paths = _filesystem_paths_report(configuration=configuration)
    summary = {
        "ok": sum(1 for report in paths.values() if report.get("ok", False)),
        "failed": sum(1 for report in paths.values() if not report.get("ok", False)),
    }
    return {"system_name": configuration.name, "paths": paths, "summary": summary}


@mcp.tool()
def check_mesoscope_bridge_tool() -> dict[str, Any]:
    """Checks whether the ScanImagePC's runAcquisition control loop is reachable for Mesoscope imaging sessions.

    Probes the MQTT command loop that the acquisition runtime uses to arm and command the ScanImage software over the
    shared broker. Use this tool during pre-flight health checks to confirm the runAcquisition function is running on
    the ScanImagePC before starting a Mesoscope imaging session (window-checking or experiment).

    Returns:
        A dictionary with ``reachable`` and ``status`` (a human-readable summary), or ``{"error": ...}`` on failure.
    """
    try:
        reachable, status = check_mesoscope_bridge()
    except Exception as exception:
        return {"error": str(exception)}
    return {"reachable": reachable, "status": status}


@mcp.tool()
def read_session_zaber_positions_tool(session_path: str) -> dict[str, Any]:
    """Loads the ZaberPositions YAML for a session.

    Args:
        session_path: The path to the session directory or its raw_data subdirectory.

    Returns:
        A dictionary with ``file_path`` and ``data`` (the serialized ZaberPositions payload), or ``{"error": ...}`` on
        failure.
    """
    session_root, error = _resolve_session_root(session_path=session_path)
    if error is not None:
        return error
    return read_yaml(
        file_path=session_root.joinpath(  # type: ignore[union-attr]
            RAW_DATA_DIRECTORY, MesoscopeRawDataFiles.ZABER_POSITIONS
        ),
        validator_cls=ZaberPositions,
    )


@mcp.tool()
def write_session_zaber_positions_tool(
    session_path: str,
    positions_payload: dict[str, Any],
    *,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Creates or replaces the ZaberPositions YAML for a session.

    Args:
        session_path: The path to the session directory or its raw_data subdirectory.
        positions_payload: The complete ZaberPositions payload to write.
        overwrite: Determines whether to overwrite an existing positions file.

    Returns:
        A dictionary with ``file_path`` and ``data`` (the validated ZaberPositions payload), or ``{"error": ...}`` on
        failure.
    """
    session_root, error = _resolve_session_root(session_path=session_path)
    if error is not None:
        return error
    return write_yaml_validated(
        file_path=session_root.joinpath(  # type: ignore[union-attr]
            RAW_DATA_DIRECTORY, MesoscopeRawDataFiles.ZABER_POSITIONS
        ),
        payload=positions_payload,
        validator_cls=ZaberPositions,
        overwrite=overwrite,
    )


@mcp.tool()
def read_session_mesoscope_positions_tool(session_path: str) -> dict[str, Any]:
    """Loads the MesoscopePositions YAML for a session.

    Args:
        session_path: The path to the session directory or its raw_data subdirectory.

    Returns:
        A dictionary with ``file_path`` and ``data`` (the serialized MesoscopePositions payload), or ``{"error": ...}``
        on failure.
    """
    session_root, error = _resolve_session_root(session_path=session_path)
    if error is not None:
        return error
    return read_yaml(
        file_path=session_root.joinpath(  # type: ignore[union-attr]
            RAW_DATA_DIRECTORY, MesoscopeRawDataFiles.MESOSCOPE_POSITIONS
        ),
        validator_cls=MesoscopePositions,
    )


@mcp.tool()
def write_session_mesoscope_positions_tool(
    session_path: str,
    positions_payload: dict[str, Any],
    *,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Creates or replaces the MesoscopePositions YAML for a session.

    Args:
        session_path: The path to the session directory or its raw_data subdirectory.
        positions_payload: The complete MesoscopePositions payload to write.
        overwrite: Determines whether to overwrite an existing positions file.

    Returns:
        A dictionary with ``file_path`` and ``data`` (the validated MesoscopePositions payload), or ``{"error": ...}``
        on failure.
    """
    session_root, error = _resolve_session_root(session_path=session_path)
    if error is not None:
        return error
    return write_yaml_validated(
        file_path=session_root.joinpath(  # type: ignore[union-attr]
            RAW_DATA_DIRECTORY, MesoscopeRawDataFiles.MESOSCOPE_POSITIONS
        ),
        payload=positions_payload,
        validator_cls=MesoscopePositions,
        overwrite=overwrite,
    )


@mcp.tool()
def read_session_system_configuration_tool(session_path: str) -> dict[str, Any]:
    """Loads the per-session snapshot of MesoscopeSystemConfiguration.

    Args:
        session_path: The path to the session directory or its raw_data subdirectory.

    Returns:
        A dictionary with ``file_path`` and ``data`` (the serialized MesoscopeSystemConfiguration snapshot), or
        ``{"error": ...}`` on failure.
    """
    session_root, error = _resolve_session_root(session_path=session_path)
    if error is not None:
        return error
    return read_yaml(
        file_path=session_root.joinpath(  # type: ignore[union-attr]
            RAW_DATA_DIRECTORY, RawDataFiles.SYSTEM_CONFIGURATION
        ),
        validator_cls=MesoscopeSystemConfiguration,
    )


@mcp.tool()
def preprocess_session_tool(session_path: str) -> str:
    """Preprocesses a session's data stored on the data acquisition system's host-machine.

    Args:
        session_path: The absolute path to the session directory to preprocess. The session must be located
            inside the data root of the data acquisition system.

    Returns:
        A success message upon completion, or an error description if preprocessing fails.
    """
    try:
        path = Path(session_path)
        system_configuration = get_system_configuration()
        data_root = get_data_root()

        # Validates that the session is stored locally.
        if not path.is_relative_to(data_root):
            return (
                f"Error: Session directory must be inside the data root of the "
                f"{system_configuration.name} data acquisition system "
                f"({data_root})."
            )

        session_data = SessionData.load(session_path=path)
        preprocess_session_data(session_data)
    except Exception as exception:
        return f"Error: {exception}"
    else:
        return f"Session preprocessed: {session_path}"


@mcp.tool()
def delete_session_tool(session_path: str, confirm_deletion: Literal["yes", "no"] | None = None) -> str:
    """Removes a session's data from all storage locations accessible to the data acquisition system.

    Important:
        This operation is irreversible and removes the session's data from every acquisition machine and every
        long-term storage destination. When ``confirm_deletion`` is omitted, the tool returns an error instead of
        deleting anything. Agentic callers should warn the user about the consequences and ask whether to proceed,
        then retry with the chosen value. A ``yes`` value performs the deletion. A ``no`` value abandons it.

    Args:
        session_path: The absolute path to the session directory to delete. The session must be located
            inside the data root of the data acquisition system.
        confirm_deletion: The policy applied to the deletion request. ``yes`` performs the deletion, ``no`` abandons
            it, and ``None`` returns an error so the caller can prompt the user.

    Returns:
        A success message upon completion, a refusal naming the accepted values when the policy is unspecified, an
        abandonment notice when the policy declines the deletion, or an error description when the deletion fails.
    """
    # Resolves the deletion policy before any storage location is touched, so an unspecified policy cannot reach the
    # purge through a falsy default.
    if confirm_deletion is None:
        return (
            f"Error: Deleting session '{session_path}' permanently removes its data from every acquisition machine "
            f"and every long-term storage destination, and cannot be undone. Specify confirm_deletion='yes' to "
            f"perform the deletion, or confirm_deletion='no' to abandon it. Ask the user which behavior they prefer "
            f"before retrying."
        )
    if confirm_deletion == "no":
        return f"Session deletion abandoned: {session_path}"

    try:
        path = Path(session_path)
        system_configuration = get_system_configuration()
        data_root = get_data_root()

        # Validates that the session is stored locally.
        if not path.is_relative_to(data_root):
            return (
                f"Error: Session directory must be inside the data root of the "
                f"{system_configuration.name} data acquisition system "
                f"({data_root})."
            )

        session_data = SessionData.load(session_path=path)
        purge_session(session_data)
    except Exception as exception:
        return f"Error: {exception}"
    else:
        return f"Session deleted: {session_path}"


@mcp.tool()
def migrate_animal_tool(source_project: str, destination_project: str, animal_id: str) -> str:
    """Transfers all sessions for an animal from one project to another.

    Args:
        source_project: The name of the project from which to migrate the data.
        destination_project: The name of the project to which to migrate the data.
        animal_id: The ID of the animal whose session data to migrate.

    Returns:
        A success message upon completion, or an error description if migration fails.
    """
    try:
        migrate_animal_between_projects(
            source_project=source_project,
            target_project=destination_project,
            animal=animal_id,
        )
    except Exception as exception:
        return f"Error: {exception}"
    else:
        return f"Animal {animal_id} migrated: {source_project} -> {destination_project}"


def _resolve_session_root(session_path: str) -> tuple[Path | None, dict[str, Any] | None]:
    """Resolves an input session path to its root directory (the parent of raw_data)."""
    path = Path(session_path)
    if not path.exists():
        return None, {"error": f"Session path does not exist: {path}"}
    if path.joinpath(RAW_DATA_DIRECTORY).is_dir():
        return path, None
    if path.name == RAW_DATA_DIRECTORY and path.is_dir():
        return path.parent, None
    return None, {"error": f"Could not locate the {RAW_DATA_DIRECTORY} directory under {path}"}


def _check_path(path: Path) -> dict[str, Any]:
    """Returns a diagnostic report for a single filesystem path."""
    report: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        report["ok"] = False
        report["error"] = "Path does not exist"
        return report
    report["is_mount"] = path.is_mount()
    write_error = probe_writable(path=path)
    report["writable"] = write_error is None
    if write_error is not None:
        report["error"] = f"Not writable: {write_error}"
    report["ok"] = report["exists"] and report["writable"]
    return report


def _check_input_file(path: Path) -> dict[str, Any]:
    """Returns a diagnostic report for a single read-only input file.

    Notes:
        Input files are consumed rather than written, so this check covers existence and read access. The mount and
        write probes applied to the storage roots would reject a valid read-only configuration file.

    Args:
        path: The path to the input file to check.

    Returns:
        A dictionary carrying the resolved path, its existence and readability flags, an ``ok`` verdict, and an
        ``error`` description when the check fails.
    """
    report: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        report["ok"] = False
        report["error"] = "File does not exist"
        return report
    report["readable"] = os.access(path, os.R_OK)
    if not report["readable"]:
        report["error"] = "Not readable"
    report["ok"] = report["readable"]
    return report


def _check_dlc_project_task(project_path: Path) -> str | None:
    """Checks that the configured DeepLabCut project is able to produce predictions accepted by preprocessing.

    Notes:
        DeepLabCut embeds the project's 'Task' field verbatim in the scorer string it appends to every prediction
        filename. Session preprocessing accepts a prediction only when that filename carries the eye-tracking token,
        so a project whose 'Task' field omits the token aborts the data transfer at the very end of a session's
        preprocessing. Reporting the mismatch here surfaces it before any session is acquired.

    Args:
        project_path: The path to the DeepLabCut project's config.yaml file.

    Returns:
        The description of why the project is unable to satisfy the eye-tracking token requirement, or None when it
        satisfies the requirement.
    """
    try:
        project = yaml.safe_load(stream=project_path.read_text(encoding="utf-8"))
    except Exception as exception:
        return f"Unable to read the DeepLabCut project configuration: {exception}"

    task = project.get("Task", "") if isinstance(project, dict) else ""
    if not isinstance(task, str) or EYE_TRACKING_PROJECT_NAME not in task:
        return (
            f"The DeepLabCut project's 'Task' field is {task!r}, which does not carry the "
            f"'{EYE_TRACKING_PROJECT_NAME}' token. DeepLabCut embeds this field in the name of every prediction "
            f"file, and session preprocessing accepts the prediction only when its name carries the token, so "
            f"preprocessing would abort the transfer to long-term storage. Point this path at a DeepLabCut project "
            f"whose 'Task' field carries the '{EYE_TRACKING_PROJECT_NAME}' token."
        )
    return None


def _filesystem_paths_report(configuration: MesoscopeSystemConfiguration) -> dict[str, Any]:
    """Builds a per-path diagnostic report for the filesystem configuration of the Mesoscope-VR system.

    Notes:
        Long-term storage destinations whose root is left unset are reported as not configured rather than as errors,
        since configuring them is optional. The optional input files follow the same convention, so a host that runs
        without stored camera configurations or without face-camera inference still reports a healthy filesystem. The
        mesoscope acquisition directory is required by this acquisition system, so an unset value there is reported as
        both not configured and not ok.

    Args:
        configuration: The Mesoscope-VR system configuration whose filesystem paths are reported on.

    Returns:
        A dictionary mapping each configuration path name to its diagnostic report.
    """
    filesystem = configuration.filesystem
    # The local data root is owned by the Sollertia platform, not the Mesoscope-VR filesystem configuration, so it
    # is resolved separately and reported as not configured when the platform data root has not been set.
    try:
        data_root_report = _check_path(path=get_data_root())
    except FileNotFoundError as exception:
        data_root_report = {"path": "", "exists": False, "ok": False, "error": str(exception)}

    # An unset path resolves to the current directory, which exists and is usually writable, so it is rejected here
    # instead of reaching the write probe and passing it.
    if filesystem.mesoscope_directory == Path():
        mesoscope_report: dict[str, Any] = {
            "path": str(filesystem.mesoscope_directory),
            "configured": False,
            "ok": False,
            "error": (
                "The filesystem.mesoscope_directory field is unset. Set it to the Mesoscope acquisition mount inside "
                "the configuration file written by the 'sle mesoscope configure system' command."
            ),
        }
    else:
        mesoscope_report = _check_path(path=filesystem.mesoscope_directory)
        mesoscope_report["configured"] = True

    paths: dict[str, Any] = {
        "data_root": data_root_report,
        "mesoscope_directory": mesoscope_report,
    }
    for destination_name, destination_root in filesystem.storage_directories.items():
        report_key = f"storage_directory:{destination_name}"
        if destination_root == Path():
            paths[report_key] = {"path": str(destination_root), "configured": False, "ok": True}
            continue
        report = _check_path(path=destination_root)
        report["configured"] = True
        paths[report_key] = report

    # Optional read-only input files. Each one is consumed by a feature the host can leave switched off, so an unset
    # path reports as not configured. A set path that fails to resolve surfaces here rather than at the point of use,
    # which for the DeepLabCut project is the end of a session's preprocessing.
    input_files: tuple[tuple[str, Path], ...] = (
        ("face_camera_configuration", configuration.cameras.face_camera_configuration_path),
        ("body_camera_configuration", configuration.cameras.body_camera_configuration_path),
        ("dlc_project", configuration.video_tracking.dlc_project_path),
    )
    for report_key, file_path in input_files:
        if file_path == Path():
            paths[report_key] = {"path": str(file_path), "configured": False, "ok": True}
            continue
        report = _check_input_file(path=file_path)
        report["configured"] = True
        paths[report_key] = report

    return paths


def _diff_genicam_configurations(stored: GenicamConfiguration, live: GenicamConfiguration) -> dict[str, Any]:
    """Builds a structured diff between a stored and a live GenICam camera configuration.

    Notes:
        A node that SFNC multiplexes behind a selector holds one value per selector combination, so it contributes
        one entry per combination rather than a single entry. Each combination is therefore compared on its own, and
        every entry in the report carries the selector values that address the instance it describes.

    Args:
        stored: The configuration loaded from the stored .yaml file.
        live: The configuration dumped from the connected camera.

    Returns:
        A dictionary describing the camera-identity match, per-node-instance value mismatches, and the node
        instances present in only one of the two configurations. It also carries an overall ``match`` flag that is
        True only when the camera identities match and every stored node instance is present on the live camera with
        the stored value.
    """
    identity_match = (
        stored.camera_model == live.camera_model and stored.camera_serial_number == live.camera_serial_number
    )

    # Keys each node by its name together with its selector values, because a selector-addressed node contributes
    # one entry per selector combination and keying by the name alone would keep only the last of those entries.
    stored_nodes = {(node.name, tuple(sorted(node.selectors.items()))): node.value for node in stored.nodes}
    live_nodes = {(node.name, tuple(sorted(node.selectors.items()))): node.value for node in live.nodes}

    value_mismatches = [
        {
            "name": name,
            "selectors": dict(selectors),
            "stored": stored_nodes[(name, selectors)],
            "live": live_nodes[(name, selectors)],
        }
        for name, selectors in sorted(stored_nodes.keys() & live_nodes.keys())
        if stored_nodes[(name, selectors)] != live_nodes[(name, selectors)]
    ]
    nodes_only_in_stored = [
        {"name": name, "selectors": dict(selectors)}
        for name, selectors in sorted(stored_nodes.keys() - live_nodes.keys())
    ]
    nodes_only_in_live = [
        {"name": name, "selectors": dict(selectors)}
        for name, selectors in sorted(live_nodes.keys() - stored_nodes.keys())
    ]

    return {
        "match": identity_match and not value_mismatches and not nodes_only_in_stored,
        "identity_match": identity_match,
        "camera_model": {"stored": stored.camera_model, "live": live.camera_model},
        "camera_serial_number": {"stored": stored.camera_serial_number, "live": live.camera_serial_number},
        "value_mismatches": value_mismatches,
        "nodes_only_in_stored": nodes_only_in_stored,
        "nodes_only_in_live": nodes_only_in_live,
    }


def _verify_single_camera(camera_index: int, configuration_path: Path) -> dict[str, Any]:
    """Verifies a single camera's live GenICam configuration against its stored configuration .yaml file.

    Args:
        camera_index: The index of the Harvester-managed camera to connect to and dump the live configuration from.
        configuration_path: The path to the stored GenICam configuration .yaml file. An unset (empty) path means the
            camera has no associated stored configuration.

    Returns:
        A dictionary with ``configured`` and, when a stored configuration is present, either an ``error`` describing
        why verification could not complete or the structured diff produced by ``_diff_genicam_configurations``.
    """
    if configuration_path == Path():
        return {"configured": False}
    if not configuration_path.exists():
        return {"configured": True, "error": f"Stored camera configuration file not found: {configuration_path}"}
    try:
        stored = GenicamConfiguration.from_yaml(file_path=configuration_path)
    except Exception as exception:
        return {"configured": True, "error": f"Failed to load stored camera configuration: {exception}"}

    try:  # pragma: no cover
        live = read_camera_configuration(camera_index=camera_index)
    except Exception as exception:  # pragma: no cover
        return {"configured": True, "error": f"Failed to read live camera configuration: {exception}"}

    return {"configured": True, **_diff_genicam_configurations(stored=stored, live=live)}  # pragma: no cover
