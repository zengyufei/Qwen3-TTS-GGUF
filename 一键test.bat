@echo off
chcp 65001 >nul
echo [3/3] 正在启动工具...

call venv\Scripts\activate
python 43-Inference-Base.py
pause