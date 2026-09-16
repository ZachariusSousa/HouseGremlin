from __future__ import annotations

import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: check_python_runtime.py EXECUTABLE MAJOR MINOR",
            file=sys.stderr,
        )
        return 2

    executable, major, minor = sys.argv[1:]
    result = subprocess.run(
        [
            executable,
            "-c",
            (
                "import sys; "
                f"raise SystemExit(sys.version_info[:2] != ({int(major)}, {int(minor)}))"
            ),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
