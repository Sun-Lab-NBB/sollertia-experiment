# Claude Code Instructions

## Session start behavior

At the beginning of each coding session, before making any code changes, you should build a comprehensive
understanding of the codebase by invoking the `/explore-codebase` skill.

This ensures you:
- Understand the project architecture before modifying code
- Follow existing patterns and conventions
- Don't introduce inconsistencies or break integrations

## Style guide compliance

You MUST invoke the appropriate `automation:*` style skill before performing ANY of the following tasks:

| Task                              | Skill to Invoke              |
|-----------------------------------|------------------------------|
| Writing or modifying Python code  | `automation:python-style`    |
| Writing or modifying README files | `automation:readme-style`    |
| Writing git commit messages       | `automation:commit`          |
| Writing or modifying skill files  | `automation:skill-design`    |
| Modifying pyproject.toml          | `automation:pyproject-style` |
| Modifying tox.ini                 | `automation:tox-config`      |
| Modifying Sphinx documentation    | `automation:api-docs`        |

This is non-negotiable. Each skill contains verification checklists that you MUST complete before submitting any work.
Failure to invoke the appropriate skill results in style violations.

## Cross-referenced library verification

Sollertia platform projects often depend on other `ataraxis-*` or `sollertia-*` libraries. These libraries may be stored
locally in the same parent directory that holds this project's repository root.

**Before writing code that interacts with a cross-referenced library, you MUST:**

1. **Check for local version**: Look for the library in the parent directory (e.g., `../ataraxis-video-system/`,
   `../sollertia-shared-assets/`).

2. **Compare versions**: If a local copy exists, compare its version against the latest release or main branch on
   GitHub:
   - Read the local `pyproject.toml` to get the current version
   - Use `gh api repos/Sun-Lab-NBB/{repo-name}/releases/latest` to check the latest release
   - Alternatively, check the main branch version on GitHub

3. **Handle version mismatches**: If the local version differs from the latest release or main branch, notify the user
   with the following options:
   - **Use online version**: Fetch documentation and API details from the GitHub repository
   - **Update local copy**: The user will pull the latest changes locally before proceeding

4. **Proceed with correct source**: Use whichever version the user selects as the authoritative reference for API
   usage, patterns, and documentation.

**Why this matters**: Skills and documentation may reference outdated APIs. Always verify against the actual library
state to prevent integration errors.

## Available skills

The sollertia marketplace ships two plugins that target this library directly: the system-agnostic `experiment` core
plugin and the `mesoscope` plugin (Mesoscope-VR system-specific skills, layered on `experiment`). Both are backed by the
`sollertia-experiment` MCP server (`sle mcp`). The ataraxis marketplace ships the `automation` plugin used across all
Sollertia platform repositories. Low-level hardware work also draws on the `video`, `communication`, and
`microcontroller` plugins, and configuration authoring draws on the `assets` plugin (see Acquisition System
Configuration below).

| Skill                                         | Description                                                          |
|-----------------------------------------------|----------------------------------------------------------------------|
| `automation:explore-codebase`                 | Perform in-depth codebase exploration at session start               |
| `automation:python-style`                     | Apply Sollertia platform Python conventions (REQUIRED for .py edits) |
| `automation:readme-style`                     | Apply Sollertia platform README conventions (REQUIRED for READMEs)   |
| `automation:commit`                           | Draft Sollertia platform style-compliant commit messages             |
| `automation:pyproject-style`                  | Apply Sollertia platform pyproject.toml conventions                  |
| `automation:tox-config`                       | Apply Sollertia platform tox.ini conventions                         |
| `automation:api-docs`                         | Apply Sollertia platform Sphinx documentation conventions            |
| `automation:project-layout`                   | Apply Sollertia platform directory structure conventions             |
| `automation:skill-design`                     | Generate, update, and verify skill files and this CLAUDE.md          |
| `automation:audit-facts`                      | Audit documentation against source code for factual accuracy         |
| `automation:audit-style`                      | Audit files against the applicable style skill checklists            |
| `experiment:pipeline`                         | Orchestrate the end-to-end experiment lifecycle                      |
| `experiment:system-design-pipeline`           | Orchestrate building a new acquisition system end-to-end             |
| `experiment:acquisition-system-design`        | Design a new acquisition system (config, bindings, runtime)          |
| `experiment:acquisition-system-runtime`       | Runtime pattern: per-mode logic, state machine, dispatch             |
| `experiment:acquisition-system-setup`         | Discover and verify connected acquisition hardware                   |
| `experiment:library-extension`                | Extension seams for a new acquisition system across sle and slmc     |
| `experiment:system-health-check`              | Pre-flight checks of configuration, mounts, and hardware             |
| `experiment:zaber-interface`                  | Implement Zaber motor interfaces and binding classes                 |
| `experiment:microcontroller-interface`        | Paired Module + ModuleInterface registry and conventions             |
| `experiment:vr-driver-interface`              | VR task driver, Unity MQTT contract, trial decomposition             |
| `experiment:data-management`                  | Preprocess, migrate, and delete session data via `sle mcp`           |
| `experiment:google-sheets-processing`         | Implement SurgeryLog / WaterLog Google Sheets processors             |
| `experiment:experiment-mcp-environment-setup` | Diagnose `sle mcp` server connectivity issues                        |
| `mesoscope:mesoscope-vr`                      | Mesoscope-VR hardware inventory, configuration, and bindings         |
| `mesoscope:mesoscope-vr-runtime`              | Mesoscope-VR state machine, orchestrator, UIs, and CLI               |
| `mesoscope:mesoscope-vr-snapshots`            | Read/write per-session Zaber and Mesoscope position snapshots        |
| `mesoscope:mesoscope-vr-session-schema`       | Mesoscope-VR session descriptor and hardware-state field schema      |
| `mesoscope:mesoscope-vr-experiment-schema`    | Mesoscope-VR experiment configuration and trial-class field schema   |

## MCP server

The library ships one MCP server, started with `sle mcp` and selecting its transport through `-t/--transport`
(`stdio` by default, `streamable-http` otherwise). It exposes two tool sets, the hardware-agnostic tools in
`interfaces/get_tools.py` and the Mesoscope-VR tools in `interfaces/mesoscope_vr_tools.py`. Neither set mirrors its
`sle` CLI counterpart, and the README lists every tool with its purpose.

`set_zaber_device_setting_tool` and `delete_session_tool` refuse to act until the caller passes an explicit `confirm`
or `confirm_deletion` value. You MUST warn the user about the consequences and obtain a decision before retrying
either tool with `yes`.

`interfaces/mcp_server.py` discovers tool modules by their `*_tools.py` filename suffix and imports each one, so a new
acquisition system registers its tools by adding `interfaces/{system}_tools.py` and needs no edit to the server module.
The CLI side carries no equivalent discovery, so that system's command group reaches the top-level `sle` group only
after one import and one `add_command()` call are added to `_register_subcommands()` in `interfaces/entry_points.py`.
The server deliberately omits the assets, video, and communication tools, which the `slsa mcp`, `axvs mcp`, and
`axci mcp` servers of those dependencies serve instead.

## Acquisition system configuration

Hardware discovery and configuration authoring are owned by different skills. You MUST invoke the appropriate skill
before helping users interact with the acquisition system.

**For hardware discovery and health checks**, use the `experiment:acquisition-system-setup` and
`experiment:system-health-check` skills. These drive this library's `sle get` commands together with the `video` and
`communication` MCP servers, and the `assets` plugin does NOT expose hardware-discovery tools. Invoke them when users
want to:
- Discover hardware (cameras, microcontrollers, Zaber motors, MQTT broker)
- Verify hardware connectivity and storage mounts before running experiments
- Troubleshoot hardware connectivity issues

Example triggers: "What cameras are connected?", "Check if the MQTT broker is running", "Verify my system
configuration".

**For configuration authoring**, use the appropriate `assets:*` skill from the `assets` plugin (backed by the
`slsa mcp` server), which reads, writes, and validates the shared configuration and metadata YAMLs. For Mesoscope-VR
hardware and calibration parameters, also consult the `mesoscope:mesoscope-vr` skill. Invoke these when users want to:
- Set up or configure an acquisition system
- Change system parameters (ports, calibration values, thresholds)

Example triggers: "Set up the mesoscope system", "Change the lick threshold".

## Project context

This is **sollertia-experiment**, the data acquisition and preprocessing runtime of the Sollertia platform. Every
Sollertia acquisition system runs in Virtual Reality, presenting a Unity task in the linear infinite corridor. The
library manages these systems and is designed to be extended with new ones. Currently, sollertia-experiment manages
the **Mesoscope-VR** two-photon imaging system, which combines brain imaging with virtual reality behavioral tasks.

### Key areas

| Directory                                | Purpose                                                  |
|------------------------------------------|----------------------------------------------------------|
| `src/sollertia_experiment/interfaces/`   | The `sle` CLI groups, the MCP server, and its tools      |
| `src/sollertia_experiment/mesoscope_vr/` | Mesoscope-VR system implementation (current system)      |
| `src/sollertia_experiment/cross_system/` | Cross-system utilities shared by all acquisition systems |
| `src/sollertia_experiment/vr_task/`      | VR task driver: Unity MQTT coupling, trial decomposition |
| `assets/mesoscope_vr/`                   | MATLAB assets deployed to the ScanImagePC, not packaged  |

### Architecture

- A single `sle` CLI entry point delegates to two command groups, a general, hardware-agnostic discovery group
  (`sle get`) and a per-system group that combines configuration, acquisition, and data management for one system
  (`sle mesoscope` for the Mesoscope-VR system), alongside the `sle mcp` command that starts the MCP server
- Hardware abstraction via binding classes (Zaber motors, cameras, microcontrollers)
- Shared memory IPC for GUI-runtime communication
- Session-based data management with distributed storage

### Code standards

- MyPy strict mode with full type annotations
- Google-style docstrings
- 120 character line limit
- See `automation:python-style` for complete conventions

### Workflow guidance

**Adding hardware to mesoscope-vr:** (see `experiment:acquisition-system-design` and `mesoscope:mesoscope-vr`)

1. Add or extend the per-subsystem configuration dataclass in `mesoscope_vr/system.py`
2. Implement binding classes in `sollertia-experiment`
3. Integrate the binding classes with the `MesoscopeVRSystem` lifecycle in `mesoscope_vr/system_controller.py`

For low-level camera hardware implementation, use the `video:camera-interface` skill.

For PC-side microcontroller hardware implementation, use the `experiment:microcontroller-interface` skill (the
registry of paired Module + ModuleInterface classes). For the underlying AXCI base API, use the
`communication:microcontroller-interface` skill.

For Zaber motor configuration, use the `experiment:zaber-interface` skill and follow the existing patterns in
`cross_system/zaber_bindings.py`.

**Adding hardware bindings (general):**

1. For shared hardware (microcontrollers), add `ModuleInterface` subclasses to `cross_system/module_interfaces.py`
2. For system-specific hardware, add wrapper classes to the system's `binding_classes.py`
3. Follow existing patterns: wrapper classes that manage device lifecycle (`connect()`, `start()`, `stop()`)
4. Use the system's own configuration dataclasses for hardware parameters (`mesoscope_vr/system.py`)

**Modifying CLI commands:** (see `mesoscope:mesoscope-vr-runtime`)

1. Identify the appropriate CLI module: `get.py` for general, hardware-agnostic discovery commands (`sle get`), or
   `mesoscope_vr.py` for Mesoscope-VR-specific commands (`sle mesoscope`, covering `configure`, `maintain`,
   `check-bridge`, `preprocess`, `delete`, `migrate`, and the `run` command group with its `window-checking`,
   `lick-training`, `run-training`, and `experiment` subcommands)
2. Add Click-decorated command functions following existing patterns
3. Import logic functions from the relevant acquisition system package
4. Register commands with the appropriate Click group. The `get` and `mesoscope` groups reach the top-level `sle`
   group through two explicit imports and two `add_command()` calls inside `_register_subcommands()` in
   `entry_points.py`, so a third group requires editing that function

**Modifying sollertia-shared-assets (session records and registries):**

sollertia-shared-assets (`../sollertia-shared-assets/`) owns the session descriptor, hardware state, experiment
configuration, and raw data classes, together with the registries in `registries.py` that key them by session type and
by acquisition system. Use the `assets:library-extension` skill for the registry extension path and the other
`assets:*` skills for authoring. System configuration is owned by this repository, in
`cross_system/system_configuration.py` and each system's `system.py`.

**Modifying sollertia-micro-controllers (hardware modules):**

Changes require updates in `sollertia-micro-controllers` (`../sollertia-micro-controllers/`) for firmware and
`sollertia-experiment` for the PC interface. Use the `experiment:microcontroller-interface` skill for the paired
Module + ModuleInterface registry and conventions, the `communication:microcontroller-interface` skill for the AXCI
base API, and the `microcontroller:firmware-module` skill for the firmware side.

**Managing session data (preprocess, migrate, delete):**

Use the `experiment:data-management` skill, which drives the `preprocess`, `migrate`, and `delete` operations exposed
by the `sle mesoscope` CLI and the `sle mcp` server.

**Adding a new acquisition system:**

Invoke `experiment:library-extension` first for the catalog of seams a new system touches across this library and
sollertia-micro-controllers, then `experiment:system-design-pipeline` for the build phase order.
