// Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// State management
let videoStream = null;
let currentImage = null;
let isCameraActive = false;

// DOM elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const previewImage = document.getElementById('previewImage');
const cameraArea = document.getElementById('cameraArea');
const cameraOverlay = document.getElementById('cameraOverlay');

// Buttons
const captureBtn = document.getElementById('captureBtn');
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const scanBtn = document.getElementById('scanBtn');
const recaptureBtn = document.getElementById('recaptureBtn');

const statusMessage = document.getElementById('statusMessage');

// Result Areas
const resultsContainer = document.getElementById('resultsContainer'); 
const alternativesContainer = document.getElementById('alternativesContainer'); 
const extractedTextDiv = document.getElementById('extractedText');

// Pages
const scanPage = document.getElementById('scanPage');
const searchPage = document.getElementById('searchPage');
const historyPage = document.getElementById('historyPage');
const pageTitle = document.getElementById('pageTitle');
const sidebarItems = document.querySelectorAll('.sidebar-item');

// Search
const manualSearchInput = document.getElementById('manualSearchInput');
const manualSearchResults = document.getElementById('manualSearchResults');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkSystemStatus();
    loadHistory();
    
    // Handle hash navigation
    const hash = window.location.hash.substring(1);
    if (hash && ['scan', 'search', 'history'].includes(hash)) {
        showPage(hash);
    }
});

function setupEventListeners() {
    // Navigation
    sidebarItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const page = item.dataset.page;
            showPage(page);
            window.location.hash = page;
        });
    });

    // Camera Interaction (Click area to start)
    cameraArea.addEventListener('click', (e) => {
        if (!isCameraActive && !currentImage && !e.target.closest('button')) {
            startCamera();
        }
    });

    captureBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        capturePhoto();
    });

    // Re-capture Button Logic
    recaptureBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        startCamera(); // Restart camera
    });

    // File Upload
    uploadBtn.addEventListener('click', () => {
        // If camera is active, stop it before opening file dialog
        if (isCameraActive) {
            stopCamera();
        }
        fileInput.click();
    });
    fileInput.addEventListener('change', handleFileSelect);

    // Scan Process
    scanBtn.addEventListener('click', processImage);

    // Manual Search
    if (manualSearchInput) {
        manualSearchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') performManualSearch(manualSearchInput.value);
        });
    }
}

function showPage(pageName) {
    [scanPage, searchPage, historyPage].forEach(p => p.style.display = 'none');
    
    switch(pageName) {
        case 'scan':
            scanPage.style.display = 'grid';
            pageTitle.textContent = 'Book Scanner';
            break;
        case 'search':
            searchPage.style.display = 'block';
            pageTitle.textContent = 'Search Books';
            break;
        case 'history':
            historyPage.style.display = 'block';
            pageTitle.textContent = 'Scan History';
            loadHistory();
            break;
    }

    sidebarItems.forEach(item => {
        if (item.dataset.page === pageName) item.classList.add('active');
        else item.classList.remove('active');
    });
}

async function checkSystemStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/status`);
        const data = await response.json();
        const statusContent = document.getElementById('statusContent');
        
        if (data.database_loaded) {
            statusContent.innerHTML = `
                <div>✓ Database Ready</div>
                <div>📚 ${data.stats.total_books.toLocaleString()} Books</div>
            `;
            statusContent.style.color = '#4ade80';
        } else {
            statusContent.innerHTML = `⚠ Database Error`;
            statusContent.style.color = '#f87171';
        }
    } catch (e) {
        console.error(e);
    }
}

// ------------------------------------------------------------------
// Camera & Image Logic
// ------------------------------------------------------------------

async function startCamera() {
    try {
        // High resolution request
        videoStream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } } 
        });
        video.srcObject = videoStream;
        video.classList.remove('hidden');
        
        // UI Updates for Camera Mode
        cameraOverlay.classList.add('hidden');
        previewImage.classList.add('hidden'); // Ensure preview is hidden
        currentImage = null; // Clear current image
        
        // Button States
        captureBtn.classList.remove('hidden'); // Show Capture
        // IMPORTANT: Ensure upload button is visible
        uploadBtn.classList.remove('hidden');
        
        recaptureBtn.classList.add('hidden');  // Hide Re-capture
        scanBtn.classList.add('hidden');       // Hide Scan
        
        isCameraActive = true;
        statusMessage.textContent = "Camera active. Click 'Capture' to take photo.";
        
    } catch (error) {
        alert('Could not access camera: ' + error.message);
    }
}

function capturePhoto() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    canvas.toBlob(blob => {
        handleImageSelection(blob, URL.createObjectURL(blob));
        stopCamera(false); // Stop stream but keep UI ready
    }, 'image/jpeg', 0.9);
}

function stopCamera(resetUI = true) {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    video.classList.add('hidden');
    isCameraActive = false;
    
    if (resetUI) {
        cameraOverlay.classList.remove('hidden');
        captureBtn.classList.add('hidden');
        // Ensure upload button stays visible
        uploadBtn.classList.remove('hidden');
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        handleImageSelection(file, URL.createObjectURL(file));
    }
}

function handleImageSelection(blob, url) {
    currentImage = blob;
    previewImage.src = url;
    previewImage.classList.remove('hidden');
    
    // UI Updates for Captured Mode
    cameraOverlay.classList.add('hidden');
    
    captureBtn.classList.add('hidden');     // Hide Capture
    // Ensure upload button stays visible
    uploadBtn.classList.remove('hidden');
    
    recaptureBtn.classList.remove('hidden'); // Show Re-capture
    scanBtn.classList.remove('hidden');      // Show Process Scan
    
    statusMessage.textContent = "Image ready. Click 'Process Scan' or 'Re-capture'.";
}

// ------------------------------------------------------------------
// Processing & Results
// ------------------------------------------------------------------

async function processImage() {
    if (!currentImage) return;
    
    showLoading(true);
    
    try {
        const formData = new FormData();
        formData.append('image', currentImage);
        
        const response = await fetch(`${API_BASE_URL}/scan`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Scan failed');
        }
        
        const data = await response.json();
        
        // Update Preview with Annotated Image
        if (data.annotated_image) {
            previewImage.src = `data:image/jpeg;base64,${data.annotated_image}`;
        }
        
        // Debug Text
        extractedTextDiv.style.display = 'block';
        extractedTextDiv.textContent = `OCR Text: "${data.extracted_text}"`;
        
        // SPLIT DISPLAY LOGIC
        if (data.results && data.results.length > 0) {
            // 1. Top Result -> RED BOX
            displayMainResult(data.results[0]);
            
            // 2. Alternatives -> GREEN BOX
            displayAlternatives(data.results.slice(1));
            
            // Save to history
            addToHistory(data.results[0]);
        } else {
            resultsContainer.innerHTML = `
                <div class="empty-state">
                    <div style="font-size:30px">🤔</div>
                    <div class="empty-state-text">No Matches Found</div>
                </div>
            `;
            alternativesContainer.innerHTML = '';
        }
        
    } catch (error) {
        alert(error.message);
    } finally {
        showLoading(false);
    }
}

function displayMainResult(book) {
    const imageHtml = book.has_local_image && book.image_data
        ? `<img src="data:image/jpeg;base64,${book.image_data}" style="width:100px; height:150px; object-fit:cover; border-radius:8px; box-shadow:0 4px 6px rgba(0,0,0,0.1);">`
        : `<div style="width:100px; height:150px; background:#eee; display:flex; align-items:center; justify-content:center; border-radius:8px;">📖</div>`;

    resultsContainer.innerHTML = `
        <div class="result-card highlight">
            <div class="result-header">
                <div style="display:flex; gap:16px; align-items:start;">
                    ${imageHtml}
                    <div>
                        <div class="book-title" style="font-size:24px;">${book.title}</div>
                        <div style="color:var(--text-medium); font-weight:600; margin-top:4px;">by ${book.author}</div>
                        <div style="margin-top:8px;">
                            <span style="background:#dbeafe; color:#1e40af; padding:4px 8px; border-radius:4px; font-size:12px; font-weight:bold;">${book.match_score}% Match</span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="result-details">
                <div class="detail-row">
                    <span class="detail-label">Category</span>
                    <span class="detail-value">${book.category}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Matched On</span>
                    <span class="detail-value">${book.matched_on.join(', ')}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">ASIN</span>
                    <span class="detail-value">${book.amazon_index}</span>
                </div>
            </div>
        </div>
    `;
}

function displayAlternatives(books) {
    if (!books || books.length === 0) {
        alternativesContainer.innerHTML = '<div style="color:#aaa; text-align:center; padding:10px;">No other close matches.</div>';
        return;
    }

    alternativesContainer.innerHTML = books.map((book, index) => `
        <div class="recent-item" onclick="promoteAlternative(${index})">
            <div style="overflow:hidden; white-space:nowrap; text-overflow:ellipsis; margin-right:10px;">
                ${book.title} <span style="color:#888; font-weight:normal;">- ${book.author}</span>
            </div>
            <div class="confidence-badge">${book.match_score}%</div>
        </div>
    `).join('');
    
    window.currentAlternatives = books;
}

window.promoteAlternative = function(index) {
    if (window.currentAlternatives && window.currentAlternatives[index]) {
        const selected = window.currentAlternatives[index];
        displayMainResult(selected);
    }
};

window.clearAllResults = function() {
    resultsContainer.innerHTML = '<div class="empty-state"><div class="empty-state-text">Cleared</div></div>';
    alternativesContainer.innerHTML = '';
    previewImage.src = '';
    previewImage.classList.add('hidden');
    scanBtn.classList.add('hidden');
    recaptureBtn.classList.add('hidden');
    // uploadBtn is always visible now
    cameraOverlay.classList.remove('hidden');
    currentImage = null;
    extractedTextDiv.style.display = 'none';
    statusMessage.textContent = '';
};

// ------------------------------------------------------------------
// Search & History
// ------------------------------------------------------------------

async function performManualSearch(query) {
    if (!query) return;
    
    manualSearchResults.innerHTML = '<div style="text-align:center; padding:20px;">Searching...</div>';
    
    try {
        const response = await fetch(`${API_BASE_URL}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        const data = await response.json();
        
        if (data.results.length === 0) {
            manualSearchResults.innerHTML = '<div class="empty-state"><div class="empty-state-text">No matches found</div></div>';
        } else {
            manualSearchResults.innerHTML = data.results.map(book => `
                <div class="result-card">
                    <div class="result-header">
                        <div class="book-title">${book.title}</div>
                        <div style="font-weight:bold; color:var(--accent-blue)">${book.match_score}%</div>
                    </div>
                    <div class="result-details">
                        <div class="detail-row"><span class="detail-label">Author</span><span>${book.author}</span></div>
                        <div class="detail-row"><span class="detail-label">Category</span><span>${book.category}</span></div>
                    </div>
                </div>
            `).join('');
        }
    } catch (e) {
        manualSearchResults.innerHTML = 'Error searching.';
    }
}

function addToHistory(book) {
    let history = JSON.parse(localStorage.getItem('scanHistory') || '[]');
    history.unshift({
        title: book.title,
        author: book.author,
        date: new Date().toLocaleDateString()
    });
    history = history.slice(0, 20);
    localStorage.setItem('scanHistory', JSON.stringify(history));
}

function loadHistory() {
    const historyList = document.getElementById('historyList');
    if (!historyList) return;
    
    const history = JSON.parse(localStorage.getItem('scanHistory') || '[]');
    
    if (history.length === 0) {
        historyList.innerHTML = '<div class="empty-state"><div class="empty-state-text">No history yet</div></div>';
        return;
    }
    
    historyList.innerHTML = history.map(item => `
        <div class="history-item">
            <div class="history-item-info">
                <div class="history-item-title">${item.title}</div>
                <div class="history-item-meta">
                    <span class="history-item-date">${item.date}</span>
                    <span class="history-item-author">${item.author}</span>
                </div>
            </div>
        </div>
    `).join('');
}

function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    overlay.style.display = show ? 'flex' : 'none';
}