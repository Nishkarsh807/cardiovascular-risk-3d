@echo off
echo ========================================================
echo   CardioVision 3D - Push to GitHub Repository
echo ========================================================
echo.
set /p REPO_URL="Enter your remote GitHub repository URL (e.g., https://github.com/yourname/cardiovascular-risk-3d.git): "

if "%REPO_URL%"=="" (
    echo Error: No repository URL entered.
    pause
    exit /b 1
)

echo.
echo Adding remote origin...
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo.
echo Pushing to main branch...
git push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================================
    echo   Successfully pushed CardioVision 3D to GitHub!
    echo ========================================================
) else (
    echo.
    echo Push failed. Please check your GitHub credentials or repository permissions.
)

pause
