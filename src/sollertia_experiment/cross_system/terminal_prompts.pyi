from typing import Any
from collections.abc import (
    Callable as Callable,
    Sequence,
)

import questionary
from prompt_toolkit.validation import Validator as Validator

_AFFIRMATIVE_RESPONSES: frozenset[str]
_NEGATIVE_RESPONSES: frozenset[str]

def wait_for_enter(message: str = "Press Enter to continue") -> None: ...
def request_confirmation(message: str, *, default: bool = False) -> bool: ...
def request_required_confirmation(message: str) -> bool: ...
def request_text(
    message: str,
    *,
    default: str = "",
    multiline: bool = False,
    validate: Callable[[str], bool | str] | Validator | type[Validator] | None = None,
) -> str: ...
def request_selection(message: str, choices: Sequence[questionary.Choice | str]) -> Any: ...
def _validate_confirmation_response(response: str) -> bool | str: ...
def _flush_input_buffer() -> None: ...
