"""Provides the health checks that verify the filesystem paths and the video-tracking project declared in the
Mesoscope-VR system configuration.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any
from pathlib import Path

import yaml
from sollertia_shared_assets import get_data_root

from ..cross_system import probe_writable
from .data_preprocessing import EYE_TRACKING_PROJECT_NAME

if TYPE_CHECKING:
    from .system import MesoscopeSystemConfiguration


def build_filesystem_paths_report(configuration: MesoscopeSystemConfiguration) -> dict[str, Any]:
    """Builds a per-path diagnostic report for the filesystem configuration of the Mesoscope-VR system.

    Notes:
        Long-term storage destinations whose root is left unset are reported as not configured rather than as errors,
        since configuring them is optional. The optional input files follow the same convention, so a host that runs
        without stored camera configurations or without face-camera inference still reports a healthy filesystem. The
        mesoscope acquisition directory is required by this acquisition system, so an unset value there is reported as
        both not configured and not ok.

    Args:
        configuration: The Mesoscope-VR system configuration whose filesystem paths the report covers.

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


def check_dlc_project_task(project_path: Path) -> str | None:
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


def _check_path(path: Path) -> dict[str, Any]:
    """Returns a diagnostic report for a single filesystem path.

    Args:
        path: The path to the directory to check.

    Returns:
        A dictionary carrying the resolved path, its existence flag and, for a path that exists, its mount and
        writability flags, together with an ``ok`` verdict and an ``error`` description when the check fails.
    """
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
