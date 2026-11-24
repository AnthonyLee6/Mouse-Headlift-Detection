@echo off
set "BASE=%~dp0"
set "PY=python"
set "NAME=User"
if not "%~1"=="" set "NAME=%~1"

echo Starting event annotation GUI  ^|  annotator = %NAME%
"%PY%" "%BASE%event_annotate_gui.py" --all --annotator "%NAME%"
if errorlevel 1 pause
