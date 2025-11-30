# 📚 BookLens - Smart Book Scanner

A web-based book scanning application that uses OCR (Optical Character Recognition) and a custom CRNN model to identify books from cover images. Simply scan or upload a book cover, and BookLens will extract text and match it against a database of over 200,000 books.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **📷 Camera Scanning** - Use your device camera to scan book covers in real-time
- **📤 Image Upload** - Upload existing book cover images for recognition
- **🔍 Smart OCR** - Combines Tesseract OCR with a custom CRNN deep learning model
- **📖 Book Database** - Search against 200,000+ books with fuzzy matching
- **🎯 High Accuracy** - Noise filtering and intelligent text region detection
- **📜 Scan History** - Keep track of your previously scanned books
- **🔎 Manual Search** - Search books by title, author, or keywords

## 🛠️ Tech Stack

- **Backend**: Flask, Python
- **OCR Engine**: Tesseract OCR + Custom CRNN (PyTorch)
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Image Processing**: OpenCV, PIL
- **Database**: Pandas (CSV-based)

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+** - [Download Python](https://www.python.org/downloads/)
- **Tesseract OCR** - Required for text extraction

### Installing Tesseract OCR

<details>
<summary><b>Windows</b></summary>

1. Download the installer from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer (recommended path: `C:\Program Files\Tesseract-OCR\`)
3. Add Tesseract to your PATH, or the app will auto-detect common installation paths

</details>

<details>
<summary><b>macOS</b></summary>

```bash
brew install tesseract
```

</details>

<details>
<summary><b>Ubuntu/Debian</b></summary>

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

</details>

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/BookLens.git
cd BookLens
```

### 2. Extract the Dataset

The book cover dataset is provided as `Dataset.zip`. Extract it to create the required folder structure:

```bash
# On Windows (PowerShell)
Expand-Archive -Path Dataset.zip -DestinationPath .

# On macOS/Linux
unzip Dataset.zip
```

After extraction, your folder structure should look like this:

```
BookLens/
├── Book_Cover_Dataset/
│   ├── 224x224/                    # Book cover images
│   │   ├── 0002005018.jpg
│   │   ├── 0060973129.jpg
│   │   └── ... (thousands of images)
│   ├── book30-listing-test.csv
│   ├── book30-listing-train.csv
│   ├── book32-listing.csv
│   ├── bookcover30-labels-test.txt
│   └── bookcover30-labels-train.txt
├── static/
│   ├── script.js
│   └── styles.css
├── templates/
│   └── index.html
├── venv/                           # Created during setup
├── .gitignore
├── app.py
├── best_ocr_model.pth              # Pre-trained CRNN model
├── Dataset.zip
├── requirements.txt
├── setup.bat                       # Windows setup script
└── setup.sh                        # macOS/Linux setup script
```

### 3. Run the Setup Script

<details>
<summary><b>Windows</b></summary>

```cmd
setup.bat
```

</details>

<details>
<summary><b>macOS/Linux</b></summary>

```bash
chmod +x setup.sh
./setup.sh
```

</details>

The setup script will:
- ✅ Check Python version
- ✅ Create a virtual environment
- ✅ Install all dependencies
- ✅ Verify Tesseract installation
- ✅ Check dataset files

### 4. Start the Application

```bash
# Activate virtual environment (if not already activated)
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# Run the Flask app
python app.py
```

### 5. Open in Browser

Navigate to: **http://localhost:5000**

---

## 📖 Manual Setup (Alternative)

If you prefer manual setup or the scripts don't work:

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Extract Dataset.zip to create Book_Cover_Dataset folder

# 5. Run the application
python app.py
```

---

## 📦 Requirements

Create a `requirements.txt` file with the following dependencies:

```txt
flask>=2.0.0
flask-cors>=3.0.0
pandas>=1.3.0
opencv-python>=4.5.0
numpy>=1.21.0
Pillow>=8.0.0
pytesseract>=0.3.8
torch>=1.9.0
torchvision>=0.10.0
```

---

## 🎮 Usage Guide

### Scanning a Book

1. **Open Camera**: Click on the camera area to start your device camera
2. **Position Book**: Hold the book cover in front of the camera
3. **Capture**: Click the "Capture" button when ready
4. **Process**: Click "Process Scan" to analyze the image
5. **View Results**: See the matched book with confidence score

### Uploading an Image

1. Click **"Upload Image"** button
2. Select a book cover image from your device
3. Click **"Process Scan"** to analyze

### Manual Search

1. Navigate to the **Search** tab in the sidebar
2. Enter book title, author, or keywords
3. Press Enter to search
4. Click on any result to view details

---

## 🔧 Configuration

### Tesseract Path (Windows)

If Tesseract is installed in a non-standard location, update `app.py`:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Your\Custom\Path\tesseract.exe'
```

### Running on a Different Port

```bash
# In app.py, change the last line:
app.run(debug=True, host='0.0.0.0', port=8080)
```

### Enabling GPU Acceleration

If you have a CUDA-compatible GPU, PyTorch will automatically use it. To verify:

```python
import torch
print(torch.cuda.is_available())  # Should print True
```

---

## 📁 Project Structure

```
BookLens/
├── app.py                  # Main Flask application & OCR engine
├── best_ocr_model.pth      # Pre-trained CRNN model weights
├── requirements.txt        # Python dependencies
├── setup.bat              # Windows setup script
├── setup.sh               # Unix setup script
│
├── Book_Cover_Dataset/    # Dataset (extracted from Dataset.zip)
│   ├── 224x224/          # Book cover images (224x224 pixels)
│   ├── book30-listing-test.csv
│   ├── book30-listing-train.csv
│   └── book32-listing.csv
│
├── static/
│   ├── script.js          # Frontend JavaScript
│   └── styles.css         # Stylesheet
│
└── templates/
    └── index.html         # Main HTML template
```

---

## 🐛 Troubleshooting

### "Tesseract not found" Error

**Solution**: Ensure Tesseract is installed and added to PATH, or manually set the path in `app.py`.

### "No module named 'cv2'" Error

**Solution**: 
```bash
pip install opencv-python
```

### "CUDA out of memory" Error

**Solution**: The model will automatically fall back to CPU. You can also force CPU usage:
```python
# In app.py, change:
self.device = 'cpu'
```

### Database Not Loading

**Solution**: Ensure `Dataset.zip` is extracted correctly and the `Book_Cover_Dataset` folder contains CSV files.

### Camera Not Working

**Solution**: 
- Ensure you're accessing via `localhost` or `HTTPS`
- Grant camera permissions in your browser
- Try a different browser (Chrome recommended)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - Open source OCR engine
- [Book Cover Dataset](https://github.com/uchidalab/book-dataset) - Book cover image dataset
- [PyTorch](https://pytorch.org/) - Deep learning framework

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

<p align="center">Made with ❤️ for book lovers</p>
