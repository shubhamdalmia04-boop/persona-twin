@echo off
rem Builds PersonaTwin-windows.zip (one PersonaTwin.exe inside). Needs Python 3.12 on THIS computer only.
cd /d "%~dp0"
py -3.12 build_exe.py
if errorlevel 1 python build_exe.py
echo.
echo The finished zip is in the "dist" folder.
pause
