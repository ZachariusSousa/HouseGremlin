from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "Scripts" / "check_python_runtime.py"


class PythonRuntimeCheckerTests(unittest.TestCase):
    def test_accepts_an_executable_with_the_requested_runtime(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                sys.executable,
                str(sys.version_info.major),
                str(sys.version_info.minor),
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_an_executable_with_a_different_runtime(self) -> None:
        wrong_minor = sys.version_info.minor - 1
        result = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                sys.executable,
                str(sys.version_info.major),
                str(wrong_minor),
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 1, result.stderr)


if __name__ == "__main__":
    unittest.main()
