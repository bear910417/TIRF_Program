@echo off
start  chrome.exe --new-tab http://127.0.0.1:8042/
cd .
call .venv\Scripts\activate.bat
@echo on
python Aoi_viewer_server.py
PAUSE