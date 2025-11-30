"""
SMART OCR SYSTEM - Improved Noise Filtering + Better Search

Improvements:
1. Book region detection using edge detection
2. Higher confidence thresholds for text detection
3. Text filtering to remove noise (random characters, numbers)
4. Focus on larger, more prominent text regions
5. IMPROVED SEARCH: Partial matching, single word queries, fuzzy search
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
# NOISE FILTERING UTILITIES
# ============================================================================

def is_valid_text(text):
    """
    Filter out noise text that's unlikely to be book title/author.
    Returns True if text appears to be valid book-related text.
    """
    if not text or len(text) < 2:
        return False
    
    text = text.strip()
    
    # Too short (likely noise)
    if len(text) < 2:
        return False
    
    # Too many numbers (likely serial numbers, dates picked up from background)
    digit_ratio = sum(c.isdigit() for c in text) / len(text)
    if digit_ratio > 0.5 and len(text) > 3:
        return False
    
    # Random character sequences (no vowels = likely noise)
    vowels = set('aeiouAEIOU')
    if len(text) > 3:
        vowel_count = sum(1 for c in text if c in vowels)
        if vowel_count == 0:
            return False
    
    # Filter out common noise patterns
    noise_patterns = [
        r'^[0-9]+$',           # Pure numbers
        r'^[^a-zA-Z]+$',       # No letters at all
        r'^[a-zA-Z]$',         # Single letter
        r'^[0-9]{4,}',         # Long number sequences (years, codes)
        r'^\W+$',              # Only special characters
        r'^[A-Z]{1,2}[0-9]+',  # Codes like "A123", "FL1817"
        r'^[0-9]+[A-Z]{1,2}$', # Codes like "1817FL"
    ]
    
    for pattern in noise_patterns:
        if re.match(pattern, text):
            return False
    
    # Gibberish detection: too many consonant clusters
    consonant_cluster = re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{4,}', text)
    if consonant_cluster and len(text) < 8:
        return False
    
    return True


def filter_text_results(results, min_confidence=40):
    """
    Filter OCR results to keep only valid text with good confidence.
    """
    filtered = []
    for r in results:
        if r['confidence'] >= min_confidence and is_valid_text(r['text']):
            filtered.append(r)
    return filtered


def find_dominant_text_region(boxes, image_shape):
    """
    Find the main text region (likely the book cover) by analyzing
    spatial distribution of detected text boxes.
    """
    if not boxes:
        return None
    
    img_h, img_w = image_shape[:2]
    img_center_x = img_w / 2
    img_center_y = img_h / 2
    
    scored_boxes = []
    
    for box_data in boxes:
        x, y, w, h = box_data['box']
        
        box_center_x = x + w / 2
        box_center_y = y + h / 2
        
        dist_from_center = np.sqrt(
            ((box_center_x - img_center_x) / img_w) ** 2 +
            ((box_center_y - img_center_y) / img_h) ** 2
        )
        
        size_score = (w * h) / (img_w * img_h)
        center_score = 1 - dist_from_center
        total_score = (size_score * 0.4) + (center_score * 0.4) + (box_data['confidence'] / 100 * 0.2)
        
        scored_boxes.append({
            **box_data,
            'score': total_score
        })
    
    return scored_boxes


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
# SMART OCR ENGINE
# ============================================================================

class SmartOCR:
    def __init__(self, model_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_loaded = False
        
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
        else:
            print(f"⚠️ Model not found at {model_path}")
    
    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def detect_book_region(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        img_area = image.shape[0] * image.shape[1]
        best_rect = None
        best_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < img_area * 0.1 or area > img_area * 0.95:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) >= 4 and area > best_area:
                best_area = area
                best_rect = cv2.boundingRect(contour)
        
        return best_rect
    
    def detect_text_boxes(self, image, book_region=None):
        processed = self.preprocess_image(image)
        custom_config = r'--oem 3 --psm 11'
        data = pytesseract.image_to_data(processed, config=custom_config, output_type=Output.DICT)
        
        boxes = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            conf = int(data['conf'][i])
            text = data['text'][i].strip()
            
            if conf > 35 and len(text) > 0:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                if w > 15 and h > 12:
                    if book_region:
                        bx, by, bw, bh = book_region
                        box_center_x = x + w / 2
                        box_center_y = y + h / 2
                        margin = 50
                        if not (bx - margin <= box_center_x <= bx + bw + margin and
                                by - margin <= box_center_y <= by + bh + margin):
                            continue
                    
                    boxes.append({
                        'box': (x, y, w, h),
                        'tesseract_text': text,
                        'confidence': conf
                    })
        
        return boxes
    
    def recognize_text_in_box(self, image, box):
        x, y, w, h = box
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return None
        
        try:
            pil_img = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(tensor)
                pred_indices = output.argmax(dim=2).squeeze(0)
                
                chars = []
                prev = None
                for idx in pred_indices:
                    idx = idx.item()
                    if idx != 0 and idx != prev and idx in self.idx_to_char:
                        chars.append(self.idx_to_char[idx])
                    prev = idx
                
                text = ''.join(chars).strip()
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
        print("📖 Attempting to detect book region...")
        book_region = self.detect_book_region(image)
        if book_region:
            print(f"   Found book region: {book_region}")
        else:
            print("   No distinct book region found, using full image")
        
        print("🔍 Detecting text regions with Tesseract...")
        boxes = self.detect_text_boxes(image, book_region)
        print(f"   Found {len(boxes)} text regions (after initial filtering)")
        
        if not boxes:
            return {
                'text': '',
                'words': [],
                'annotated_image': image,
                'success': False,
                'message': 'No text detected',
                'stats': {'custom_model': 0, 'tesseract': 0, 'total_words': 0}
            }
        
        scored_boxes = find_dominant_text_region(boxes, image.shape)
        scored_boxes.sort(key=lambda x: x['score'], reverse=True)
        
        results = []
        annotated_image = image.copy()
        
        for i, box_data in enumerate(scored_boxes):
            box = box_data['box']
            x, y, w, h = box
            
            recognized_text = None
            method_used = 'tesseract'
            confidence = box_data['confidence']
            
            if self.model_loaded:
                result = self.recognize_text_in_box(image, box)
                if result and result['text'] and len(result['text']) > 0:
                    recognized_text = result['text']
                    confidence = result['confidence']
                    method_used = 'custom_model'
            
            if not recognized_text:
                recognized_text = box_data['tesseract_text']
                method_used = 'tesseract'
            
            if recognized_text and len(recognized_text) > 0:
                results.append({
                    'text': recognized_text,
                    'box': box,
                    'confidence': confidence,
                    'method': method_used,
                    'score': box_data.get('score', 0)
                })
                
                color = (0, 255, 0) if method_used == 'custom_model' else (255, 165, 0)
                cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color, 2)
        
        print(f"🧹 Filtering results (before: {len(results)} words)...")
        filtered_results = filter_text_results(results, min_confidence=40)
        print(f"   After filtering: {len(filtered_results)} words")
        
        filtered_out = [r for r in results if r not in filtered_results]
        if filtered_out:
            print(f"   Filtered out: {[r['text'] for r in filtered_out]}")
        
        filtered_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        full_text = " ".join([r['text'] for r in filtered_results])
        
        for r in filtered_results:
            x, y, w, h = r['box']
            color = (0, 255, 0) if r['method'] == 'custom_model' else (0, 200, 255)
            cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color, 3)
        
        model_count = sum(1 for r in filtered_results if r['method'] == 'custom_model')
        tesseract_count = len(filtered_results) - model_count
        
        print(f"\n📊 Recognition Stats:")
        print(f"   Custom Model: {model_count} words")
        print(f"   Tesseract: {tesseract_count} words")
        print(f"   Final text: '{full_text}'\n")
        
        return {
            'text': full_text,
            'words': filtered_results,
            'annotated_image': annotated_image,
            'success': True,
            'stats': {
                'custom_model': model_count,
                'tesseract': tesseract_count,
                'total_words': len(filtered_results),
                'filtered_out': len(results) - len(filtered_results)
            }
        }

# ============================================================================
# INITIALIZE OCR ENGINE
# ============================================================================

ocr_engine = SmartOCR(MODEL_PATH)

# ============================================================================
# BOOK DATABASE - IMPROVED SEARCH
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
            
            # Create lowercase versions for faster searching
            self.df['title_lower'] = self.df['Title'].fillna('').astype(str).str.lower()
            self.df['author_lower'] = self.df['Author'].fillna('').astype(str).str.lower()
            
            self.loaded = True
            print(f"✓ Database Ready: {len(self.df)} books")
            return True
        return False

    def get_image_path(self, filename):
        if not filename: return None
        path = os.path.join(IMAGES_DIR, filename)
        return path if os.path.exists(path) else None

    def clean_text_for_ocr(self, text):
        """Clean text for OCR matching - more strict, filters stop words"""
        if not isinstance(text, str): return []
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        words = text.split()
        return [w for w in words if w not in STOP_WORDS and len(w) > 2]
    
    def clean_text_for_search(self, text):
        """Clean text for manual search - less strict, keeps short words"""
        if not isinstance(text, str): return []
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        words = text.split()
        # Keep words with 2+ characters for search
        return [w for w in words if len(w) >= 2]

    def calculate_coverage(self, ocr_tokens, db_text):
        db_tokens = self.clean_text_for_ocr(db_text)
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

    def search(self, query_text, max_results=20):
        """
        IMPROVED SEARCH FUNCTION
        - Supports single word queries
        - Supports partial word matching (prefix search)
        - Case-insensitive
        - Fuzzy matching for typos
        """
        if not self.loaded or self.df is None: 
            return []
        
        query_text = query_text.strip().lower()
        if not query_text:
            return []
        
        print(f"🔍 Searching for: '{query_text}'")
        
        # Get search tokens (less strict - keeps short words)
        search_tokens = self.clean_text_for_search(query_text)
        
        # Also keep the original query for partial matching
        original_query = re.sub(r'[^a-zA-Z0-9\s]', '', query_text).strip()
        
        if not search_tokens and not original_query:
            return []
        
        print(f"   Search tokens: {search_tokens}")
        print(f"   Original query: '{original_query}'")
        
        results = []
        
        # METHOD 1: Direct substring/partial match (for single words like "mind", "calc")
        # This catches partial matches like "calc" -> "calculus"
        if original_query:
            mask_partial = (
                self.df['title_lower'].str.contains(original_query, case=False, na=False, regex=False) |
                self.df['author_lower'].str.contains(original_query, case=False, na=False, regex=False)
            )
            partial_matches = self.df[mask_partial]
            print(f"   Partial matches found: {len(partial_matches)}")
            
            for idx, row in partial_matches.iterrows():
                title = str(row.get('Title', ''))
                author = str(row.get('Author', ''))
                title_lower = row.get('title_lower', '')
                author_lower = row.get('author_lower', '')
                
                # Calculate match score based on how well query matches
                score = 0
                matched_on = []
                
                # Check if query appears in title
                if original_query in title_lower:
                    # Higher score if query is at the start of a word
                    words_in_title = title_lower.split()
                    for word in words_in_title:
                        if word.startswith(original_query):
                            score += 80  # Prefix match
                            break
                        elif original_query in word:
                            score += 60  # Substring match
                            break
                    matched_on.append('title')
                
                # Check if query appears in author
                if original_query in author_lower:
                    words_in_author = author_lower.split()
                    for word in words_in_author:
                        if word.startswith(original_query):
                            score += 40
                            break
                        elif original_query in word:
                            score += 30
                            break
                    matched_on.append('author')
                
                # Bonus for exact word match
                if original_query in title_lower.split():
                    score += 20
                if original_query in author_lower.split():
                    score += 10
                
                if score > 0:
                    results.append({
                        'title': title,
                        'author': author,
                        'category': row.get('Category', 'General'),
                        'amazon_index': row.get('Amazon Index', ''),
                        'match_score': min(100, score),
                        'matched_on': matched_on,
                        'row_idx': idx
                    })
        
        # METHOD 2: Token-based matching (for multi-word queries)
        if search_tokens:
            # Build regex pattern for token matching
            pattern = '|'.join([re.escape(t) for t in search_tokens])
            
            mask_tokens = (
                self.df['title_lower'].str.contains(pattern, case=False, na=False, regex=True) |
                self.df['author_lower'].str.contains(pattern, case=False, na=False, regex=True)
            )
            token_matches = self.df[mask_tokens]
            print(f"   Token matches found: {len(token_matches)}")
            
            for idx, row in token_matches.iterrows():
                # Skip if already added from partial matching
                if any(r.get('row_idx') == idx for r in results):
                    continue
                
                title = str(row.get('Title', ''))
                author = str(row.get('Author', ''))
                
                author_score = self.calculate_coverage(search_tokens, author)
                title_score = self.calculate_coverage(search_tokens, title)
                
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
                elif (title_score + author_score) > 60:
                    final_score = (title_score + author_score) / 2
                    matched_on.append('mixed')
                
                if final_score >= 25:
                    results.append({
                        'title': title,
                        'author': author,
                        'category': row.get('Category', 'General'),
                        'amazon_index': row.get('Amazon Index', ''),
                        'match_score': int(final_score),
                        'matched_on': matched_on,
                        'row_idx': idx
                    })
        
        # METHOD 3: Fuzzy matching for typos (if few results)
        if len(results) < 5 and original_query and len(original_query) >= 3:
            print(f"   Trying fuzzy matching...")
            # Sample titles for fuzzy matching (limit for performance)
            sample_size = min(10000, len(self.df))
            sample_df = self.df.sample(n=sample_size) if len(self.df) > sample_size else self.df
            
            for idx, row in sample_df.iterrows():
                if any(r.get('row_idx') == idx for r in results):
                    continue
                
                title_lower = row.get('title_lower', '')
                author_lower = row.get('author_lower', '')
                
                # Check fuzzy match on words
                title_words = title_lower.split()
                author_words = author_lower.split()
                
                fuzzy_score = 0
                matched_on = []
                
                for word in title_words:
                    if len(word) >= 3:
                        ratio = difflib.SequenceMatcher(None, original_query, word).ratio()
                        if ratio > 0.7:
                            fuzzy_score = max(fuzzy_score, int(ratio * 70))
                            if 'title' not in matched_on:
                                matched_on.append('title')
                
                for word in author_words:
                    if len(word) >= 3:
                        ratio = difflib.SequenceMatcher(None, original_query, word).ratio()
                        if ratio > 0.7:
                            fuzzy_score = max(fuzzy_score, int(ratio * 50))
                            if 'author' not in matched_on:
                                matched_on.append('author')
                
                if fuzzy_score >= 40:
                    results.append({
                        'title': str(row.get('Title', '')),
                        'author': str(row.get('Author', '')),
                        'category': row.get('Category', 'General'),
                        'amazon_index': row.get('Amazon Index', ''),
                        'match_score': fuzzy_score,
                        'matched_on': matched_on,
                        'row_idx': idx
                    })
        
        # Remove duplicates and sort by score
        seen_titles = set()
        unique_results = []
        for r in results:
            title_key = r['title'].lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_results.append(r)
        
        unique_results.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Add image data to top results
        final_results = []
        for r in unique_results[:max_results]:
            # Get image data
            row_idx = r.get('row_idx')
            if row_idx is not None and row_idx in self.df.index:
                row = self.df.loc[row_idx]
                image_path = self.get_image_path(row.get('Filename'))
                img_data = None
                if image_path:
                    try:
                        with open(image_path, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode('utf-8')
                    except: 
                        pass
                
                r['has_local_image'] = image_path is not None
                r['image_data'] = img_data
            else:
                r['has_local_image'] = False
                r['image_data'] = None
            
            # Remove internal tracking field
            if 'row_idx' in r:
                del r['row_idx']
            
            final_results.append(r)
        
        print(f"   Final results: {len(final_results)}")
        return final_results

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
        ocr_result = ocr_engine.extract_text(image)
        
        if not ocr_result['success']:
            return jsonify({'error': ocr_result['message']}), 500
        
        results = book_db.search(ocr_result['text'])
        
        _, buffer = cv2.imencode('.jpg', ocr_result['annotated_image'])
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'extracted_text': ocr_result['text'],
            'annotated_image': annotated_base64,
            'cleaned_query': " ".join(book_db.clean_text_for_search(ocr_result['text'])),
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