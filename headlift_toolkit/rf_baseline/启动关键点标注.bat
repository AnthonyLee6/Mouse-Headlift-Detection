@echo off
set "BASE=%~dp0"
set "PY=python"
set "NAME=User"
if not "%~1"=="" set "NAME=%~1"

echo Starting keypoint GUI  ^|  annotator = %NAME%
"%PY%" "%BASE%keypoint_gui.py" --annotator "%NAME%"
if errorlevel 1 pause
