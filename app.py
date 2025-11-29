"""
SMART OCR SYSTEM - Let the Model Do Its Thing!

Strategy:
1. Your custom model handles text recognition (what it's trained for)
2. Tesseract handles text detection (finding where text is)
3. OpenCV helps with preprocessing
4. Everything works together seamlessly
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output
import torch
import torch.nn as nn
from torchvision import transforms
import base64
import re
import difflib
import os
import sys

# Configure Tesseract
if sys.platform == 'win32':
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract-OCR\tesseract.exe',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"✓ Tesseract found: {path}")
            break

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'Book_Cover_Dataset')
IMAGES_DIR = os.path.join(DATASET_DIR, '224x224')
MODEL_PATH = os.path.join(BASE_DIR, 'best_ocr_model.pth')

STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
    'edition', 'vol', 'volume', 'book', 'press', 'publishers'
}

# ============================================================================
# CRNN MODEL
# ============================================================================

class CRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes, hidden_size=256):
        super(CRNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), 
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), 
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, (2, 1), 1, 0), nn.BatchNorm2d(512), nn.ReLU(),
        )
        
        self.rnn = nn.LSTM(512, hidden_size, 2, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        conv = self.cnn(x)
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv)
        return self.fc(rnn_out)

# ============================================================================
# SMART OCR ENGINE - Model Does Recognition, Tesseract Does Detection
# ============================================================================

class SmartOCR:
    """
    Division of Labor:
    - Custom Model: Text recognition (reading words)
    - Tesseract: Text detection (finding text locations)
    - OpenCV: Image preprocessing
    """
    
    def __init__(self, model_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_loaded = False
        
        # Try to load custom model
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.config = checkpoint['config']
                self.idx_to_char = checkpoint['idx_to_char']
                
                self.model = CRNN(
                    self.config['img_height'],
                    3,
                    self.config['num_classes'],
                    self.config['hidden_size']
                ).to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                self.transform = transforms.Compose([
                    transforms.Resize((self.config['img_height'], self.config['img_width'])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                self.model_loaded = True
                print(f"✅ Custom Model Ready on {self.device}")
            except Exception as e:
                print(f"⚠️ Model load failed: {e}")
                print("   → Will use Tesseract as fallback")
        else:
            print(f"⚠️ Model not found at {model_path}")
            print("   → Will use Tesseract as fallback")
    
    def preprocess_image(self, image):
        """OpenCV preprocessing for better OCR"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Binarize
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def detect_text_boxes(self, image):
        """Use Tesseract to find WHERE text is located"""
        processed = self.preprocess_image(image)
        
        custom_config = r'--oem 3 --psm 11'
        data = pytesseract.image_to_data(processed, config=custom_config, output_type=Output.DICT)
        
        boxes = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            conf = int(data['conf'][i])
            text = data['text'][i].strip()
            
            if conf > 20 and len(text) > 0:  # Low threshold, we'll recognize later
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                if w > 10 and h > 10:  # Filter tiny regions
                    boxes.append({
                        'box': (x, y, w, h),
                        'tesseract_text': text,  # Keep as reference
                        'confidence': conf
                    })
        
        return boxes
    
    def recognize_text_in_box(self, image, box):
        """Use YOUR MODEL to read text in a detected box"""
        x, y, w, h = box
        
        # Extract region with padding
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return None
        
        try:
            # Convert to PIL and transform
            pil_img = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            # Run YOUR MODEL
            with torch.no_grad():
                output = self.model(tensor)
                pred_indices = output.argmax(dim=2).squeeze(0)
                
                # CTC decode
                chars = []
                prev = None
                for idx in pred_indices:
                    idx = idx.item()
                    if idx != 0 and idx != prev and idx in self.idx_to_char:
                        chars.append(self.idx_to_char[idx])
                    prev = idx
                
                text = ''.join(chars).strip()
                
                # Calculate confidence
                probs = torch.softmax(output, dim=2)
                confidence = probs.max(dim=2)[0].mean().item() * 100
                
                return {
                    'text': text,
                    'confidence': confidence,
                    'method': 'custom_model'
                }
        except Exception as e:
            print(f"Model recognition error: {e}")
            return None
    
    def extract_text(self, image):
        """
        Main OCR pipeline:
        1. Tesseract finds text locations
        2. Your model reads the text
        3. Tesseract fills in if model fails
        """
        
        # Step 1: Find text boxes (Tesseract does this)
        print("🔍 Detecting text regions with Tesseract...")
        boxes = self.detect_text_boxes(image)
        print(f"   Found {len(boxes)} text regions")
        
        if not boxes:
            return {
                'text': '',
                'words': [],
                'success': False,
                'message': 'No text detected'
            }
        
        # Step 2: Recognize text in each box
        results = []
        annotated_image = image.copy()
        
        for i, box_data in enumerate(boxes):
            box = box_data['box']
            x, y, w, h = box
            
            recognized_text = None
            method_used = 'tesseract'
            confidence = box_data['confidence']
            
            # Try custom model first (if loaded)
            if self.model_loaded:
                result = self.recognize_text_in_box(image, box)
                
                if result and result['text'] and len(result['text']) > 0:
                    recognized_text = result['text']
                    confidence = result['confidence']
                    method_used = 'custom_model'
                    print(f"   ✓ Model: '{recognized_text}' (conf: {confidence:.1f}%)")
            
            # Fallback to Tesseract if model failed or not loaded
            if not recognized_text:
                recognized_text = box_data['tesseract_text']
                method_used = 'tesseract'
                print(f"   → Tesseract: '{recognized_text}' (conf: {confidence}%)")
            
            # Save result
            if recognized_text and len(recognized_text) > 0:
                results.append({
                    'text': recognized_text,
                    'box': box,
                    'confidence': confidence,
                    'method': method_used
                })
                
                # Draw box (color based on method)
                color = (0, 255, 0) if method_used == 'custom_model' else (255, 0, 0)
                cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color, 2)
                
                # Add label
                label = f"{recognized_text[:15]} ({method_used[0].upper()})"
                cv2.putText(annotated_image, label, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Step 3: Combine all text
        full_text = " ".join([r['text'] for r in results])
        
        # Stats
        model_count = sum(1 for r in results if r['method'] == 'custom_model')
        tesseract_count = len(results) - model_count
        
        print(f"\n📊 Recognition Stats:")
        print(f"   Custom Model: {model_count} words")
        print(f"   Tesseract: {tesseract_count} words")
        print(f"   Total: '{full_text}'\n")
        
        return {
            'text': full_text,
            'words': results,
            'annotated_image': annotated_image,
            'success': True,
            'stats': {
                'custom_model': model_count,
                'tesseract': tesseract_count,
                'total_words': len(results)
            }
        }

# ============================================================================
# INITIALIZE OCR ENGINE
# ============================================================================

ocr_engine = SmartOCR(MODEL_PATH)

# ============================================================================
# BOOK DATABASE (Unchanged)
# ============================================================================

class BookDatabase:
    def __init__(self):
        self.df = None
        self.loaded = False
    
    def load_database(self):
        print("Loading book database...")
        csv_files = ['book30-listing-test.csv', 'book30-listing-train.csv', 'book32-listing.csv']
        dfs = []
        
        for csv_file in csv_files:
            csv_path = os.path.join(DATASET_DIR, csv_file)
            if not os.path.exists(csv_path):
                continue
                
            try:
                delimiter = ','
                try:
                    with open(csv_path, 'r', encoding='latin-1') as f:
                        header = f.readline()
                        if ';' in header and header.count(';') > header.count(','):
                            delimiter = ';'
                except:
                    pass

                df = pd.read_csv(csv_path, encoding='latin-1', sep=delimiter, on_bad_lines='skip', engine='python')
                
                column_mapping = {
                    'Amazon Index (ASIN)': 'Amazon Index', 'asin': 'Amazon Index',
                    'Filename': 'Filename', 'Image URL': 'Image URL',
                    'Title': 'Title', 'Author': 'Author',
                    'Category': 'Category', 'Category ID': 'Category ID'
                }
                
                current_cols = {c.lower().strip(): c for c in df.columns}
                rename_dict = {}
                for k, v in column_mapping.items():
                    if k.lower() in current_cols:
                        rename_dict[current_cols[k.lower()]] = v
                
                df.rename(columns=rename_dict, inplace=True)
                
                if 'Title' in df.columns:
                    dfs.append(df)
                    print(f"✓ Loaded {csv_file}: {len(df)} records")
                
            except Exception as e:
                print(f"✗ Error loading {csv_file}: {e}")

        if dfs:
            self.df = pd.concat(dfs, ignore_index=True)
            if 'Filename' in self.df.columns:
                self.df.drop_duplicates(subset=['Filename'], keep='first', inplace=True)
            
            for col in ['Title', 'Author', 'Category', 'Amazon Index']:
                if col not in self.df.columns:
                    self.df[col] = ''
            
            self.loaded = True
            print(f"✓ Database Ready: {len(self.df)} books")
            return True
        return False

    def get_image_path(self, filename):
        if not filename: return None
        path = os.path.join(IMAGES_DIR, filename)
        return path if os.path.exists(path) else None

    def clean_text(self, text):
        if not isinstance(text, str): return []
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        words = text.split()
        return [w for w in words if w not in STOP_WORDS and len(w) > 2]

    def calculate_coverage(self, ocr_tokens, db_text):
        db_tokens = self.clean_text(db_text)
        if not db_tokens: return 0
        hits = 0
        for db_word in db_tokens:
            if db_word in ocr_tokens:
                hits += 1
                continue
            matches = difflib.get_close_matches(db_word, ocr_tokens, n=1, cutoff=0.8)
            if matches:
                hits += 1
        return (hits / len(db_tokens)) * 100

    def search(self, ocr_text, max_results=20):
        if not self.loaded or self.df is None: return []
        
        ocr_tokens = self.clean_text(ocr_text)
        if not ocr_tokens: return []
        
        pattern = '|'.join([re.escape(t) for t in ocr_tokens])
        mask = (
            self.df['Title'].fillna('').astype(str).str.contains(pattern, case=False, regex=True) | 
            self.df['Author'].fillna('').astype(str).str.contains(pattern, case=False, regex=True)
        )
        candidates = self.df[mask]
        
        if len(candidates) == 0:
            return []

        results = []
        for idx, row in candidates.iterrows():
            title = str(row.get('Title', ''))
            author = str(row.get('Author', ''))
            
            author_score = self.calculate_coverage(ocr_tokens, author)
            title_score = self.calculate_coverage(ocr_tokens, title)
            
            final_score = 0
            matched_on = []
            
            if author_score > 80:
                final_score += 60
                matched_on.append('author')
                if title_score > 20:
                    final_score += title_score * 0.4
                    matched_on.append('title')
                else:
                    final_score += 10 
            elif title_score > 50:
                final_score += title_score
                matched_on.append('title')
                if author_score > 40:
                    final_score += 20
                    matched_on.append('author')
            elif (title_score + author_score) > 80:
                final_score = (title_score + author_score) / 2
                matched_on.append('mixed')

            if final_score >= 30:
                image_path = self.get_image_path(row.get('Filename'))
                img_data = None
                if image_path:
                    try:
                        with open(image_path, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode('utf-8')
                    except: pass

                results.append({
                    'title': title,
                    'author': author,
                    'category': row.get('Category', 'General'),
                    'amazon_index': row.get('Amazon Index', ''),
                    'match_score': int(final_score),
                    'matched_on': matched_on,
                    'has_local_image': image_path is not None,
                    'image_data': img_data
                })
        
        results.sort(key=lambda x: x['match_score'], reverse=True)
        return results[:max_results]

    def get_stats(self):
        if not self.loaded: return {}
        return {
            'total_books': int(len(self.df)),
            'categories': int(self.df['Category'].nunique()) if 'Category' in self.df.columns else 0
        }

book_db = BookDatabase()

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    if book_db.loaded:
        return jsonify({
            'database_loaded': True,
            'model_loaded': ocr_engine.model_loaded,
            'stats': book_db.get_stats()
        })
    return jsonify({'database_loaded': False, 'model_loaded': ocr_engine.model_loaded})

@app.route('/api/scan', methods=['POST'])
def scan():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    
    try:
        # Use smart OCR (model + tesseract fallback)
        ocr_result = ocr_engine.extract_text(image)
        
        if not ocr_result['success']:
            return jsonify({'error': ocr_result['message']}), 500
        
        # Search database
        results = book_db.search(ocr_result['text'])
        
        # Encode annotated image
        _, buffer = cv2.imencode('.jpg', ocr_result['annotated_image'])
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'extracted_text': ocr_result['text'],
            'annotated_image': annotated_base64,
            'cleaned_query': " ".join(book_db.clean_text(ocr_result['text'])),
            'results': results,
            'ocr_stats': ocr_result['stats']
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def manual_search():
    data = request.get_json()
    query = data.get('query', '')
    results = book_db.search(query, max_results=20)
    return jsonify({'success': True, 'results': results})

if __name__ == '__main__':
    book_db.load_database()
    app.run(debug=True, host='0.0.0.0', port=5000)