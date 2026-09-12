from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "scripts" / "check-skippy-workload-candidate.py"


class CandidateBuildFreshnessTests(unittest.TestCase):
    def _check(self, binary: Path, build_dir: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(CHECK),
                "--candidate-binary",
                str(binary),
                "--native-build-dir",
                str(build_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_accepts_binary_linked_after_stamped_native_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            stamp = directory / ".mesh-llm-build-stamp"
            binary = directory / "skippy-server"
            stamp.touch()
            binary.touch(mode=0o755)
            os.utime(stamp, ns=(1_000_000_000, 1_000_000_000))
            os.utime(binary, ns=(2_000_000_000, 2_000_000_000))
            self.assertEqual(0, self._check(binary, directory).returncode)

    def test_rejects_binary_older_than_or_equal_to_native_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            stamp = directory / ".mesh-llm-build-stamp"
            binary = directory / "skippy-server"
            stamp.touch()
            binary.touch(mode=0o755)
            os.utime(stamp, ns=(2_000_000_000, 2_000_000_000))
            for binary_time in (1_000_000_000, 2_000_000_000):
                with self.subTest(binary_time=binary_time):
                    os.utime(binary, ns=(binary_time, binary_time))
                    result = self._check(binary, directory)
                    self.assertNotEqual(0, result.returncode)
                    self.assertIn("candidate executable predates", result.stderr)

    def test_rejects_missing_executable_or_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            binary = directory / "skippy-server"
            self.assertIn("candidate executable is missing", self._check(binary, directory).stderr)
            binary.touch(mode=0o755)
            self.assertIn("native build stamp is missing", self._check(binary, directory).stderr)


if __name__ == "__main__":
    unittest.main()
