#!/usr/bin/env bash
set -e
python -m pytest -q
python examples/test_run_modes.py
