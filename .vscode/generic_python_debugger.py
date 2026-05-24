from __future__ import annotations

import argparse
import os
import runpy
import shlex
import sys
from pathlib import Path


def _split_args(raw_args: str) -> list[str]:
    if not raw_args.strip():
        return []
    return shlex.split(raw_args, posix=os.name != "nt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic Python debug launcher")
    parser.add_argument(
        "--kind",
        choices=("script", "module"),
        required=True,
        help="Target type to execute",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Script path or module name to execute",
    )
    parser.add_argument(
        "--cwd",
        default="",
        help="Working directory to switch to before running the target",
    )
    parser.add_argument(
        "--args",
        default="",
        help="Command-line arguments as a single quoted string",
    )
    parsed = parser.parse_args()

    if parsed.cwd:
        os.chdir(parsed.cwd)

    forwarded_args = _split_args(parsed.args)
    sys.argv = [parsed.target, *forwarded_args]

    if parsed.kind == "script":
        script_path = Path(parsed.target)
        if not script_path.is_absolute():
            script_path = Path.cwd() / script_path
        runpy.run_path(str(script_path), run_name="__main__")
    else:
        runpy.run_module(parsed.target, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
