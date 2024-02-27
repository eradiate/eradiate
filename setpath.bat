@echo off

REM ***************************************************************
REM * This script adds Eradiate to the current path on Windows.
REM * It assumes that Mitsuba is compiled within a subdirectory
REM * named 'build'.
REM ***************************************************************

set ERADIATE_SOURCE_DIR=%~dp0
set PATH=%ERADIATE_SOURCE_DIR%ext\mitsuba\build\Release;%PATH%
set PYTHONPATH=%ERADIATE_SOURCE_DIR%ext\mitsuba\build\Release\python;%PYTHONPATH%
