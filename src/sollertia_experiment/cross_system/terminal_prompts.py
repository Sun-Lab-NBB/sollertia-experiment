"""Provides terminal prompt helpers that discard buffered input and require the Enter key to submit every response."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import questionary
from prompt_toolkit.input.defaults import create_input
from prompt_toolkit.input.typeahead import clear_typeahead

try:
    import termios
except ImportError:  # The termios module is only available on POSIX platforms.
    termios = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Sequence


def wait_for_enter(message: str = "Press Enter to continue.") -> None:
    """Blocks runtime execution until the user presses the Enter key.

    This replaces the single-key "press any key" prompt, which advanced on any keystroke and could consume key presses
    buffered ahead of time. Requiring the Enter key ensures the user deliberately resumes the runtime.

    Args:
        message: The message displayed to the user while the runtime waits for the Enter key.
    """
    _flush_input_buffer()
    input(message)


def request_confirmation(message: str, *, default: bool = False) -> bool:
    """Prompts the user to confirm or decline an action, requiring the Enter key to submit the response.

    Args:
        message: The yes-or-no question presented to the user.
        default: The response used when the user submits an empty answer.

    Returns:
        True if the user confirmed the action, False if the user declined it.
    """
    _flush_input_buffer()
    response: bool = questionary.confirm(message=message, default=default, auto_enter=False).unsafe_ask()
    return response


def request_text(message: str, *, default: str = "", multiline: bool = False, validate: Any = None) -> str:
    """Prompts the user to enter free-form text, requiring the Enter key to submit the response.

    Args:
        message: The instruction presented to the user.
        default: The text pre-filled into the response field.
        multiline: Determines whether the user can enter multiple lines of text before submitting the response.
        validate: An optional questionary validator applied to the response before it is accepted.

    Returns:
        The text entered by the user.
    """
    _flush_input_buffer()
    response: str = questionary.text(
        message=message, default=default, multiline=multiline, validate=validate
    ).unsafe_ask()
    return response


def request_selection(message: str, choices: Sequence[questionary.Choice | str]) -> Any:
    """Prompts the user to select one option from a list, requiring the Enter key to submit the selection.

    Args:
        message: The instruction presented to the user.
        choices: The options the user can select from, provided as questionary choices or plain strings.

    Returns:
        The value associated with the option selected by the user.
    """
    _flush_input_buffer()
    response: Any = questionary.select(message=message, choices=list(choices)).unsafe_ask()
    return response


def _flush_input_buffer() -> None:
    """Discards any terminal input keystrokes buffered before this point.

    This clears both the prompt_toolkit type-ahead buffer, which stores keystrokes read while a previous prompt was
    active, and the operating system terminal buffer, which stores keystrokes entered while no prompt was active.
    Clearing both prevents buffered key presses from being consumed by the next prompt.
    """
    # Skips flushing when the standard input is not an interactive terminal, as there is no buffer to clear and the
    # termios call below would fail.
    if not sys.stdin.isatty():
        return

    # Clears the prompt_toolkit type-ahead buffer keyed to the standard input file descriptor.
    clear_typeahead(create_input())

    # Discards keystrokes received by the operating system terminal but not yet read by the application.
    if termios is not None:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
