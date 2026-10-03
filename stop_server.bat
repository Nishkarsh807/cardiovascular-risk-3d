@echo off
echo Stopping any running CardioVision 3D server...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Terminating PID %%a on port 8000...
    taskkill /F /PID %%a >nul 2>&1
)
echo Server stopped.
pause
