#!/bin/bash

echo "======================================"
echo "Book Scanner Setup Script"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1)
if [ $? -eq 0 ]; then
    echo "✓ $python_version found"
else
    echo "✗ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Check for Tesseract
echo ""
echo "Checking for Tesseract OCR..."
if command -v tesseract &> /dev/null; then
    tesseract_version=$(tesseract --version 2>&1 | head -n 1)
    echo "✓ $tesseract_version found"
else
    echo "✗ Tesseract OCR not found!"
    echo ""
    echo "Please install Tesseract:"
    echo "  macOS:   brew install tesseract"
    echo "  Ubuntu:  sudo apt-get install tesseract-ocr"
    echo "  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
fi

# Check for dataset
echo ""
echo "Checking for dataset..."
if [ -d "Book_Cover_Dataset" ]; then
    echo "✓ Book_Cover_Dataset folder found"
    
    # Check for CSV files
    csv_count=$(ls Book_Cover_Dataset/*.csv 2>/dev/null | wc -l)
    if [ $csv_count -gt 0 ]; then
        echo "✓ Found $csv_count CSV file(s)"
    else
        echo "✗ No CSV files found in Book_Cover_Dataset/"
    fi
    
    # Check for images
    if [ -d "Book_Cover_Dataset/224x224" ]; then
        image_count=$(ls Book_Cover_Dataset/224x224/*.jpg 2>/dev/null | wc -l)
        echo "✓ Found $image_count book cover images"
    else
        echo "✗ Book_Cover_Dataset/224x224/ folder not found"
    fi
else
    echo "✗ Book_Cover_Dataset folder not found!"
    echo ""
    echo "Please create the folder structure:"
    echo "  book-scanner/"
    echo "  └── Book_Cover_Dataset/"
    echo "      ├── 224x224/              (book cover images)"
    echo "      ├── book30-listing-test.csv"
    echo "      ├── book30-listing-train.csv"
    echo "      └── book32-listing.csv"
fi

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "To start the application:"
echo "  1. Activate virtual environment (if not already):"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the Flask app:"
echo "     python app.py"
echo ""
echo "  3. Open browser to:"
echo "     http://localhost:5000"
echo ""
echo "======================================"
