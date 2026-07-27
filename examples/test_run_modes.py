from __future__ import annotations
import os
import sys
import subprocess

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_SCRIPT = os.path.join(CURRENT_DIR, "run_single_case.py")
CONFIG_FILE = os.path.join(CURRENT_DIR, "parameters_smoke_test.txt")

MODES = ["exact", "rank1", "adaptive", "hybrid"]


def run_one(mode: str):
    cmd = [
        sys.executable,
        RUN_SCRIPT,
        "--config",
        CONFIG_FILE,
        "--mode",
        mode,
    ]
    print("=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)
    result = subprocess.run(cmd, check=False)
    return result.returncode


def main():
    failures = []

    for mode in MODES:
        rc = run_one(mode)
        if rc != 0:
            failures.append((mode, rc))

    print("\n" + "=" * 80)
    if failures:
        print("Some mode tests FAILED:")
        for mode, rc in failures:
            print(f"  - {mode}: return code {rc}")
        sys.exit(1)
    else:
        print("All mode tests PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()