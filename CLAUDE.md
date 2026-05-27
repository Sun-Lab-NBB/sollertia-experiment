# Claude Code Instructions

## Session Start Behavior

At the beginning of each coding session, before making any code changes, you should build a comprehensive
understanding of the codebase by invoking the `/explore-codebase` skill.

This ensures you:
- Understand the project architecture before modifying code
- Follow existing patterns and conventions
- Don't introduce inconsistencies or break integrations

## Style Guide Requirements

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

## Acquisition System Configuration

When users want to interact with the acquisition system hardware or configuration, you MUST invoke the appropriate
`assets:*` skill from the sollertia-shared-assets plugin. The plugin exposes MCP tools for hardware discovery and
configuration management.

**Invoke the appropriate assets skill when users want to:**
- Discover hardware (cameras, microcontrollers, Zaber motors, MQTT broker)
- Set up or configure an acquisition system
- Change system parameters (ports, calibration values, thresholds)
- Verify system configuration before running experiments
- Troubleshoot hardware connectivity or configuration issues

**Example triggers:**
- "What cameras are connected?"
- "Set up the mesoscope system"
- "Change the lick threshold"
- "Check if the MQTT broker is running"
- "Verify my system configuration"

## Cross-Referenced Library Verification

Sollertia platform projects often depend on other `ataraxis-*` or `sollertia-*` libraries. These libraries may be stored
locally in the same parent directory as this project (`/home/cyberaxolotl/Desktop/GitHubRepos/`).

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

## Project Context

This is **sollertia-experiment**, the data acquisition and preprocessing runtime of the Sollertia platform. The
library is designed to manage any combination of Sollertia platform data acquisition systems and can be extended to
support new systems or modified to remove existing ones. Currently, sollertia-experiment manages the **Mesoscope-VR**
two-photon imaging system, which combines brain imaging with virtual reality behavioral tasks.

### Key Areas

| Directory                                | Purpose                                                  |
|------------------------------------------|----------------------------------------------------------|
| `src/sollertia_experiment/interfaces/`   | CLI entry points (consolidated under the `sle` command)  |
| `src/sollertia_experiment/mesoscope_vr/` | Mesoscope-VR system implementation (current system)      |
| `src/sollertia_experiment/cross_system/` | Cross-system utilities shared by all acquisition systems |

### Architecture

- A single `sle` CLI entry point delegates to two layers: a general, hardware-agnostic discovery group (`sle get`)
  and a per-system group that combines configuration, acquisition, and data management for one system
  (`sle mesoscope` for the Mesoscope-VR system)
- Hardware abstraction via binding classes (Zaber motors, cameras, microcontrollers)
- Shared memory IPC for GUI-runtime communication
- Session-based data management with distributed storage

### Code Standards

- MyPy strict mode with full type annotations
- Google-style docstrings
- 120 character line limit
- See `automation:python-style` for complete conventions

### Workflow Guidance

**Adding hardware to mesoscope-vr:**

1. Add configuration dataclasses in `sollertia-shared-assets`
2. Implement binding classes in `sollertia-experiment`
3. Integrate with `data_acquisition.py` lifecycle

For low-level camera hardware implementation, use the `video:camera-interface` skill.

For low-level microcontroller hardware implementation, use the `communication:microcontroller-interface` skill.

For Zaber motor configuration, follow the existing patterns in `mesoscope_vr/zaber_bindings.py`.

**Adding hardware bindings (general):**

1. For shared hardware (microcontrollers), add `ModuleInterface` subclasses to
   `cross_system/module_interfaces.py`
2. For system-specific hardware, add wrapper classes to the system's `binding_classes.py`
3. Follow existing patterns: wrapper classes that manage device lifecycle (`connect()`, `start()`, `stop()`)
4. Use configuration dataclasses from `sollertia-shared-assets` for hardware parameters

**Modifying CLI commands:**

1. Identify the appropriate CLI module: `get.py` for general, hardware-agnostic discovery commands (`sle get`), or
   `mesoscope_vr.py` for Mesoscope-VR-specific commands (`sle mesoscope`, covering `configure`, `maintain`, `run`,
   `preprocess`, `delete`, and `migrate`)
2. Add Click-decorated command functions following existing patterns
3. Import logic functions from the relevant acquisition system package
4. Register commands with the appropriate Click group (the `get` and `mesoscope` groups are auto-registered on the
   top-level `sle` group via `entry_points.py`)

**Modifying sollertia-shared-assets (configuration dataclasses):**

Changes to system configuration require updates in `sollertia-shared-assets` (`../sollertia-shared-assets/`). Use the
`assets:*` skills from the sollertia-shared-assets plugin for guidance.

**Modifying sollertia-micro-controllers (hardware modules):**

Changes require updates in `sollertia-micro-controllers` (`../sollertia-micro-controllers/`) for firmware and
`sollertia-experiment` for the PC interface. Use the `communication:microcontroller-interface` and
`microcontroller:firmware-module` skills for guidance.
