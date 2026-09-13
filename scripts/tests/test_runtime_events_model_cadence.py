"""Keep the native runtime-event gate's real fixture executable in PR and main CI."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
RESOLVER = ROOT / "scripts/resolve-test-model-manifest.py"


class RuntimeEventModelCadenceTests(unittest.TestCase):
    """Resolve the exact model selected by the protected Linux runtime workflow."""

    def gate_inputs(self) -> dict[str, str]:
        """Read the workflow's fixture selector instead of duplicating its manifest."""
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/ci-linux-runtime-slice.yml").read_text()
        )
        step = next(step for step in workflow["jobs"]["linux_runtime"]["steps"]
                    if step.get("id") == "gate_model")
        self.assertEqual(step["uses"], "./.github/actions/restore-test-model")
        self.assertEqual(step["with"]["model_cadence"],
                         "${{ (inputs.original_event_name == 'pull_request' || "
                         "inputs.original_event_name == 'pull_request_target') "
                         "&& 'pull-request' || 'main' }}")
        return step["with"]

    def resolve(self, manifest: Path, artifact: str, cadence: str) -> subprocess.CompletedProcess:
        """Exercise the production resolver with the gate's single-file requirement."""
        return subprocess.run(
            ["python3", str(RESOLVER), str(manifest), "--artifact-id", artifact,
             "--cadence", cadence, "--require-single-file"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )

    def test_native_gate_model_resolves_for_pr_and_main(self) -> None:
        """Both event branches must resolve before the native reporter test can run."""
        inputs = self.gate_inputs()
        for cadence in ("pull-request", "main"):
            with self.subTest(cadence=cadence):
                result = self.resolve(ROOT / inputs["model_manifest"],
                                      inputs["model_artifact_id"], cadence)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["artifact_id"],
                                 inputs["model_artifact_id"])

    def test_missing_cadence_still_fails_closed(self) -> None:
        """Adding gate membership must not turn the resolver into a permissive fallback."""
        inputs = self.gate_inputs()
        source = json.loads((ROOT / inputs["model_manifest"]).read_text())
        for cadence in ("pull-request", "main"):
            with self.subTest(cadence=cadence), tempfile.TemporaryDirectory() as directory:
                manifest = json.loads(json.dumps(source))
                artifact = next(row for row in manifest["artifacts"]
                                if row["id"] == inputs["model_artifact_id"])
                artifact["cadences"] = [value for value in artifact["cadences"]
                                         if value != cadence]
                path = Path(directory) / "manifest.json"
                path.write_text(json.dumps(manifest))
                result = self.resolve(path, inputs["model_artifact_id"], cadence)
                self.assertEqual(result.returncode, 2)
                self.assertIn("is not allowed at cadence", result.stderr)

    def test_native_gate_does_not_expand_family_certification_cadences(self) -> None:
        """PR/main model retrieval must not add persistent-runner family executions."""
        family = json.loads((ROOT / "ci/llama-canary/family-certified.json").read_text())
        dense = next(row for row in family["models"] if row["family"] == "qwen3-dense")
        self.assertEqual(dense["cadences"], ["llama-bump", "manual-full", "nightly"])


if __name__ == "__main__":
    unittest.main()
