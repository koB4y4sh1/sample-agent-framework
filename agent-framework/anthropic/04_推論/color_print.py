from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Sequence
from typing import Any, TextIO

RESET = "\033[0m"
_COLOR_CODES = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_black": "\033[90m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
}
_STYLE_CODES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "underline": "\033[4m",
}
_WINDOWS_ANSI_ENABLED = False


def _enable_windows_ansi() -> None:
    global _WINDOWS_ANSI_ENABLED

    if _WINDOWS_ANSI_ENABLED or os.name != "nt":
        return

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetStdHandle(-11)
    if handle == 0:
        return

    mode = ctypes.c_uint32()
    if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
        return

    enable_virtual_terminal_processing = 0x0004
    kernel32.SetConsoleMode(handle, mode.value | enable_virtual_terminal_processing)
    _WINDOWS_ANSI_ENABLED = True


def print_color(
    *values: Any,
    color: str,
    styles: Sequence[str] | None = None,
    sep: str = " ",
    end: str = "\n",
    file: TextIO | None = None,
    flush: bool = False,
) -> None:
    output = file if file is not None else sys.stdout
    if color not in _COLOR_CODES:
        available = ", ".join(sorted(_COLOR_CODES))
        raise ValueError(f"Unsupported color: {color}. Available colors: {available}")
    if styles:
        unsupported_styles = [style for style in styles if style not in _STYLE_CODES]
        if unsupported_styles:
            available = ", ".join(sorted(_STYLE_CODES))
            invalid = ", ".join(unsupported_styles)
            raise ValueError(f"Unsupported styles: {invalid}. Available styles: {available}")

    _enable_windows_ansi()
    text = sep.join(str(value) for value in values)
    style_prefix = "".join(_STYLE_CODES[style] for style in styles or [])
    print(f"{style_prefix}{_COLOR_CODES[color]}{text}{RESET}", end=end, file=output, flush=flush)


def print_blue(*values: Any, **kwargs: Any) -> None:
    print_color(*values, color="blue", **kwargs)


def printb(*values: Any, **kwargs: Any) -> None:
    print_blue(*values, **kwargs)


def print_green(*values: Any, **kwargs: Any) -> None:
    print_color(*values, color="green", **kwargs)


def print_red(*values: Any, **kwargs: Any) -> None:
    print_color(*values, color="red", **kwargs)


def print_yellow(*values: Any, **kwargs: Any) -> None:
    print_color(*values, color="yellow", **kwargs)


def print_cyan(*values: Any, **kwargs: Any) -> None:
    print_color(*values, color="cyan", **kwargs)


def print_magenta(*values: Any, **kwargs: Any) -> None:
    print_color(*values, color="magenta", **kwargs)


def print_white(*values: Any, **kwargs: Any) -> None:
    print_color(*values, color="white", **kwargs)


def print_gray(*values: Any, **kwargs: Any) -> None:
    print_color(*values, color="bright_black", **kwargs)


def print_bright_blue(*values: Any, **kwargs: Any) -> None:
    print_color(*values, color="bright_blue", **kwargs)


def print_bright_green(*values: Any, **kwargs: Any) -> None:
    print_color(*values, color="bright_green", **kwargs)


def print_bright_red(*values: Any, **kwargs: Any) -> None:
    print_color(*values, color="bright_red", **kwargs)


def print_bright_yellow(*values: Any, **kwargs: Any) -> None:
    print_color(*values, color="bright_yellow", **kwargs)


def print_bright_magenta(*values: Any, **kwargs: Any) -> None:
    print_color(*values, color="bright_magenta", **kwargs)


def print_bright_cyan(*values: Any, **kwargs: Any) -> None:
    print_color(*values, color="bright_cyan", **kwargs)


def print_bright_white(*values: Any, **kwargs: Any) -> None:
    print_color(*values, color="bright_white", **kwargs)
