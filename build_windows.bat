@echo off
REM ---------------------------------------------------------------------------
REM Builds the standalone Windows app:  dist\SpacecraftTracker\SpacecraftTracker.exe
REM
REM Usage:   build_windows.bat gpu     (default; CUDA build, also runs on CPU-only PCs)
REM          build_windows.bat cpu     (smaller, CPU only)
REM
REM Needs on THIS machine only: Python 3.11, Git, internet. Run once.
REM Before running, copy these into this folder (they are not in Git):
REM    models\spark_yolov8n_best.pt
REM    models\sam2.1_hiera_small.pt
REM    video1.mp4   (optional test clip)
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set MODE=%1
if "%MODE%"=="" set MODE=gpu
if /I "%MODE%"=="cpu" (
    set TORCH_INDEX=https://download.pytorch.org/whl/cpu
) else (
    set TORCH_INDEX=https://download.pytorch.org/whl/cu126
)
echo Build mode: %MODE%   PyTorch index: %TORCH_INDEX%

where py >nul 2>nul || (echo ERROR: Python launcher not found. Install Python 3.11 from python.org. & goto :fail)
py -3.11 --version >nul 2>nul || (echo ERROR: Python 3.11 not installed. & goto :fail)
where git >nul 2>nul || (echo ERROR: Git not found. Install Git for Windows. & goto :fail)

if not exist models\spark_yolov8n_best.pt (echo ERROR: models\spark_yolov8n_best.pt is missing. & goto :fail)
if not exist models\sam2.1_hiera_small.pt (echo ERROR: models\sam2.1_hiera_small.pt is missing. & goto :fail)

if not exist third_party\sam2\setup.py (
    git clone https://github.com/facebookresearch/sam2.git third_party\sam2 || goto :fail
    git -C third_party\sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4 || goto :fail
)

if not exist .venv-build\Scripts\python.exe (
    py -3.11 -m venv .venv-build || goto :fail
)
call .venv-build\Scripts\activate.bat

python -m pip install --upgrade pip || goto :fail
pip install torch==2.7.1 torchvision==0.22.1 --index-url %TORCH_INDEX% || goto :fail

set SAM2_BUILD_CUDA=0
pip install third_party\sam2 || goto :fail
pip install ultralytics PySide6 opencv-python harvesters numpy pyinstaller || goto :fail

python -c "import torch; print('torch', torch.__version__, '| CUDA build:', torch.version.cuda)"

pyinstaller SpacecraftTracker.spec --noconfirm --clean || goto :fail

echo.
echo Build finished: dist\SpacecraftTracker\SpacecraftTracker.exe
echo Running the self-test of the packaged app...
dist\SpacecraftTracker\SpacecraftTracker.exe --selftest
echo Self-test exit code: %ERRORLEVEL%
pause
exit /b 0

:fail
echo.
echo BUILD FAILED. Scroll up for the first error and send it to me.
pause
exit /b 1