#!/usr/bin/env python3
"""Reject a stale statically linked candidate before an oracle comparison."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def check_candidate(binary: Path, build_dir: Path) -> None:
    stamp = build_dir / ".mesh-llm-build-stamp"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"candidate executable is missing: {binary}")
    if not stamp.is_file():
        raise RuntimeError(f"candidate native build stamp is missing: {stamp}")
    if binary.stat().st_mtime_ns <= stamp.stat().st_mtime_ns:
        raise RuntimeError(
            "candidate executable predates the stamped native ABI; "
            "rebuild skippy-server against the current LLAMA_STAGE_BUILD_DIR"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-binary", required=True, type=Path)
    parser.add_argument("--native-build-dir", required=True, type=Path)
    args = parser.parse_args()
    check_candidate(args.candidate_binary, args.native_build_dir)


if __name__ == "__main__":
    main()
