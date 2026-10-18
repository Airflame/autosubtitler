@echo off
rem Launches the GUI without a console window. Files dropped onto this .bat are added to the queue.
cd /d "%~dp0"
start "" .venv\Scripts\pythonw.exe -m autosubtitler %*
