"""Provides the consolidated 'sle' CLI entry point for the sollertia-experiment library.

The warning filter is applied at module level before any other imports to ensure deprecation warnings from
dependencies are suppressed during the import phase.
"""

import warnings as wa

wa.warn_explicit = wa.warn = lambda *_, **__: None

import click  # noqa: E402

CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.group("sle", context_settings=CONTEXT_SETTINGS)
def sle_cli() -> None:  # pragma: no cover
    """Top-level entry point for the sollertia-experiment library.

    Exposes two operational command groups: 'get' for general, hardware-agnostic acquisition system discovery, and
    'mesoscope' for configuring, running, and managing the Mesoscope-VR data acquisition system.
    """


def _register_subcommands() -> None:
    """Imports and registers all subcommand groups on the top-level 'sle' Click group.

    The imports are deferred to this helper to keep the module's import surface minimal when only metadata is
    needed by the wheel build or by tools like importlib.metadata.
    """
    from .get import get  # noqa: PLC0415
    from .mesoscope_vr import mesoscope  # noqa: PLC0415

    # noinspection PyTypeChecker
    sle_cli.add_command(cmd=get)
    # noinspection PyTypeChecker
    sle_cli.add_command(cmd=mesoscope)


_register_subcommands()
