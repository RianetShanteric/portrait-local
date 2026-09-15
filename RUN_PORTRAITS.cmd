@echo off
setlocal
chcp 65001 >nul
set "ROOT=%~dp0"
if not exist "%ROOT%.venv\Scripts\python.exe" goto setup
if not exist "%ROOT%models\deeplabv3_resnet50_coco-cd0a2569.pth" goto setup
goto run

:setup
call "%ROOT%INSTALL.cmd"
if errorlevel 1 exit /b 1

:run
"%ROOT%.venv\Scripts\python.exe" -m app.cli --root "%ROOT%" --profile balanced
pause
