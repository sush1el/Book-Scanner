@echo off
echo ======================================
echo Book Scanner Setup Script
echo ======================================
echo.

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo X Python not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do set python_version=%%i
echo + %python_version% found

REM Check if virtual environment exists
if not exist "venv\" (
    echo.
    echo Creating virtual environment...
    python -m venv venv
    echo + Virtual environment created
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo + Virtual environment activated

REM Install dependencies
echo.
echo Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo + Dependencies installed

REM Check for Tesseract
echo.
echo Checking for Tesseract OCR...
where tesseract >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('tesseract --version 2^>^&1 ^| findstr "tesseract"') do set tesseract_version=%%i
    echo + Tesseract found
) else (
    echo X Tesseract OCR not found!
    echo.
    echo Please install Tesseract from:
    echo   https://github.com/UB-Mannheim/tesseract/wiki
    echo.
    echo After installation, add to PATH or set in app.py:
    echo   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
)

REM Check for dataset
echo.
echo Checking for dataset...
if exist "Book_Cover_Dataset\" (
    echo + Book_Cover_Dataset folder found
    
    REM Check for CSV files
    dir /b Book_Cover_Dataset\*.csv >nul 2>&1
    if %errorlevel% equ 0 (
        for /f %%i in ('dir /b Book_Cover_Dataset\*.csv ^| find /c /v ""') do set csv_count=%%i
        echo + Found %csv_count% CSV file(s)
    ) else (
        echo X No CSV files found in Book_Cover_Dataset\
    )
    
    REM Check for images
    if exist "Book_Cover_Dataset\224x224\" (
        for /f %%i in ('dir /b Book_Cover_Dataset\224x224\*.jpg 2^>nul ^| find /c /v ""') do set image_count=%%i
        echo + Found %image_count% book cover images
    ) else (
        echo X Book_Cover_Dataset\224x224\ folder not found
    )
) else (
    echo X Book_Cover_Dataset folder not found!
    echo.
    echo Please create the folder structure:
    echo   book-scanner\
    echo   └── Book_Cover_Dataset\
    echo       ├── 224x224\              (book cover images)
    echo       ├── book30-listing-test.csv
    echo       ├── book30-listing-train.csv
    echo       └── book32-listing.csv
)

echo.
echo ======================================
echo Setup Complete!
echo ======================================
echo.
echo To start the application:
echo   1. Activate virtual environment (if not already):
echo      venv\Scripts\activate
echo.
echo   2. Run the Flask app:
echo      python app.py
echo.
echo   3. Open browser to:
echo      http://localhost:5000
echo.
echo ======================================
pause
