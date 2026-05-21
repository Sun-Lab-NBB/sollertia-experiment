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

    Exposes four operational command groups: 'get' for hardware discovery, 'manage' for session and storage
    operations, 'run' for data acquisition runtimes, and 'configure' for acquisition-system configuration.
    """


def _register_subcommands() -> None:
    """Imports and registers all subcommand groups on the top-level 'sle' Click group.

    The imports are deferred to this helper to keep the module's import surface minimal when only metadata is
    needed by the wheel build or by tools like importlib_metadata.
    """
    from .get import get  # noqa: PLC0415
    from .manage import manage  # noqa: PLC0415
    from .execute import run  # noqa: PLC0415
    from .configure import configure_cli  # noqa: PLC0415

    sle_cli.add_command(cmd=get)
    sle_cli.add_command(cmd=manage)
    sle_cli.add_command(cmd=run)
    sle_cli.add_command(cmd=configure_cli, name="configure")


_register_subcommands()
