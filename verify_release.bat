@echo off
python -m pytest -q
if errorlevel 1 exit /b 1
python examples\test_run_modes.py
