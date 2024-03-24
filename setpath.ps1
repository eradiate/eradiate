# ***************************************************************
# * This script adds Eradiate to the current path on Windows.
# * It assumes that Mitsuba is compiled within a subdirectory
# * named 'build'.
# ***************************************************************

$env:ERADIATE_SOURCE_DIR = Get-Location
$env:PATH = $env:ERADIATE_SOURCE_DIR + "\ext\mitsuba\build\Release" + ";" + $env:PATH
$env:PYTHONPATH = $env:ERADIATE_SOURCE_DIR + "\ext\mitsuba\build\Release\python" + ";" + $env:PYTHONPATH
