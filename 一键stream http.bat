@echo off
chcp 65001 >nul
echo [3/3] 正在启动工具...

call venv\Scripts\activate
python web_stream_service.py
pause