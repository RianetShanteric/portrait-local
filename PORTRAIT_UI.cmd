@echo off
setlocal
chcp 65001 >nul
set "ROOT=%~dp0"
if not exist "%ROOT%.venv\Scripts\pythonw.exe" goto setup
if not exist "%ROOT%models\deeplabv3_resnet50_coco-cd0a2569.pth" goto setup
"%ROOT%.venv\Scripts\python.exe" -c "import torch,cv2,customtkinter" >nul 2>&1
if errorlevel 1 goto setup
goto launch

:setup
call "%ROOT%INSTALL.cmd"
if errorlevel 1 exit /b 1

:launch
powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Process -FilePath '%ROOT%.venv\Scripts\pythonw.exe' -ArgumentList '-m','app.ui','--root','%ROOT%' -WorkingDirectory '%ROOT%'"
