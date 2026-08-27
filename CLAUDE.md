# Claude Code Instructions

## Session start behavior

At the beginning of each coding session, before making any code changes, you should build a comprehensive
understanding of the codebase by invoking the `automation:explore-codebase` skill.

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

This library depends on `ataraxis-time`, `ataraxis-base-utilities`, `ataraxis-data-structures`,
`ataraxis-transport-layer-pc`, `ataraxis-communication-interface`, `ataraxis-video-system`, and
`sollertia-shared-assets`, each pinned to an exact version in `pyproject.toml`. It also drives the
`sollertia-micro-controllers` firmware over serial and the `sollertia-virtual-reality` Unity project over MQTT. Local
clones of all of these typically live alongside this repository, in its parent directory. The external tool bindings
below are reached as subprocesses instead, so none of them is version-checked here.

**Before writing code that interacts with a cross-referenced library, you MUST:**

1. **Check for local version**: Look for the library in the parent directory (e.g.,
   `../ataraxis-communication-interface/`, `../ataraxis-video-system/`, and `../sollertia-shared-assets/`).

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
`microcontroller` plugins, and configuration authoring draws on the `assets` plugin (see Downstream library integration
below).

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
| `experiment:cli-reference`                    | Document the `sle` root, `sle mcp`, and `sle get` CLI surface        |
| `experiment:external-tool-bindings`           | Bind a tool that cannot be installed beside the stack                |
| `experiment:experiment-mcp-environment-setup` | Diagnose `sle mcp` server connectivity issues                        |
| `mesoscope:mesoscope-vr`                      | Mesoscope-VR hardware inventory, configuration, and bindings         |
| `mesoscope:mesoscope-vr-runtime`              | Mesoscope-VR state machine, orchestrator, UIs, and CLI               |
| `mesoscope:mesoscope-vr-snapshots`            | Read/write per-session Zaber and Mesoscope position snapshots        |
| `mesoscope:mesoscope-vr-session-schema`       | Mesoscope-VR session descriptor and hardware-state field schema      |
| `mesoscope:mesoscope-vr-experiment-schema`    | Mesoscope-VR experiment configuration and trial-class field schema   |
| `mesoscope:mesoscope-vr-cli-reference`        | Reference for the `sle mesoscope` commands and their options         |

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

## Downstream library integration

Hardware discovery and configuration authoring are owned by different skills. You MUST invoke the appropriate skill
before helping users interact with the acquisition system.

**For hardware discovery and health checks**, use the `experiment:acquisition-system-setup` and
`experiment:system-health-check` skills. These drive this library's `sle mcp` server and `sle get` commands together
with read-only `slsa mcp` checks and the `axvs mcp` and `axci mcp` servers of the `video` and `communication` plugins.
The `assets` plugin does NOT expose hardware-discovery tools. Invoke them when users want to:
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

## Companion library synchronization

The companion `sollertia-micro-controllers` (`../sollertia-micro-controllers/`) C++ library is the firmware counterpart
to this library, and parts of this codebase track it in lockstep. Any change to a firmware `Module` subclass's parameter
structure, status codes, command codes, controller IDs, keepalive interval, or per-target module layout requires a
matching change here. That change touches the system-agnostic `ModuleInterface` wrappers in
`cross_system/module_interfaces.py`, the per-system binding classes in `mesoscope_vr/binding_classes.py`, and the
`MesoscopeMicroControllers` configuration dataclass in `mesoscope_vr/system.py`. The
`experiment:microcontroller-interface` skill owns the paired Module + ModuleInterface list, and
`microcontroller:firmware-module` covers the firmware side.

## External tool bindings

An acquisition system may need a tool this stack cannot host, because its runtime, its dependency pins, its license, or
its own launcher forbids installing or driving it beside the stack. Such a tool is bound rather than registered. It is
invoked as a subprocess, it is never imported, and what it contributes is the artifact it writes into the session tree
rather than an API. The `experiment:external-tool-bindings` skill owns the convention, the admission test that decides
whether a dependency registers or binds, and the workflow for adding one. Invoke it before wiring any tool that cannot
share this environment.

This library carries the producer half of every binding. A binding declares the tool's address in identity fields on
the acquisition system's configuration, and launches only when every one of them is set. It resolves that address at
call time rather than at import time. It returns silently when the host has not configured the tool, and logs a warning
and returns when the tool's input is missing, so preprocessing completes in both cases. The consumer half lives in
`sollertia-forgery`, where a donated locator finds the artifact and decides whether the dependent job is possible for
that session. `forging:processing-input-format` owns that half.

One binding exists today. Mesoscope-VR preprocessing invokes `slvt infer` from `sollertia-video-tracking` through
`conda run`, because DeepLabCut supports only Python 3.10 to 3.12 and the numpy 1.x series, against this stack's
Python 3.14 and numpy 2. That inference runs alongside the other preprocessing stages and is joined before the
checksum, so a failed run aborts the transfer and retains the local session copy. The `mesoscope:mesoscope-vr` skill
documents the invocation and its configuration fields.

## Distribution model

The package ships to PyPI as `sollertia-experiment` and installs the `sle` CLI. Its Claude Code skills ship separately,
through the [sollertia](https://github.com/Sun-Lab-NBB/sollertia) marketplace, in its `experiment` and `mesoscope`
plugins, and the `experiment` plugin also registers the `sle mcp` server. An agent asked to add or change a skill edits
`sollertia/plugins/<plugin>/skills/<skill>/SKILL.md` in that repository rather than this one, and bumps that plugin's
`version` in its `.claude-plugin/plugin.json` exactly once per branch.

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

`experiment:vr-driver-interface` owns the host side of the Unity coupling alone. The Unity side lives in the
`sollertia-virtual-reality` project and is owned by the unity plugin, through `unity:mqtt-contract` for the topic
constants, `unity:play-mode` and `unity:task-scenes` for the editor bridge and scene activation, and
`unity:gimbl-framework` for the VR framework itself.

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

**Modifying CLI commands:** (see `experiment:acquisition-system-setup` for the six `sle get` commands,
`mesoscope:mesoscope-vr-cli-reference` for the `sle mesoscope` command and option surface, and
`experiment:library-extension` for the `_register_subcommands()` registration seam)

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

See `## Companion library synchronization` above for the lockstep contract and the owning skills.

**Managing session data (preprocess, migrate, delete):**

Use the `experiment:data-management` skill, which drives the `preprocess`, `migrate`, and `delete` operations exposed
by the `sle mesoscope` CLI and the `sle mcp` server.

**Adding a new acquisition system:**

Invoke `experiment:library-extension` first for the catalog of seams a new system touches across this library and
sollertia-micro-controllers, then `experiment:system-design-pipeline` for the build phase order.
