"""
Flask API for Fire Detection with Explainable AI.
Allows users to upload images and receive fire predictions with visual explanations.
"""

from flask import Flask, request, jsonify, send_file, render_template_string
from werkzeug.utils import secure_filename
import os
import json
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

# Try to import GradCAM version, fall back to standard, or use demo if torch fails
XAI_TYPE = "Unknown"
XAIModule = None
xai = None

try:
    from explainable_ai_gradcam import FireDetectionXAIGradCAM as XAIModule
    XAI_TYPE = "GradCAM"
    print("✅ GradCAM XAI module loaded successfully")
except Exception as e:
    # If GradCAM fails (e.g., torch DLL error), try standard version
    print(f"⚠️  GradCAM import failed: {str(e)[:60]}...")
    try:
        from explainable_ai import FireDetectionXAI as XAIModule
        XAI_TYPE = "Standard"
        print("✅ Standard XAI module loaded successfully")
    except Exception as e2:
        # If both fail, set to demo mode
        print(f"⚠️  Standard XAI also failed: {str(e2)[:60]}...")
        print("🎬 Running in DEMO MODE (simulated results)")
        XAI_TYPE = "DEMO"
        XAIModule = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Initialize XAI module only if available
print(f"Initializing XAI module ({XAI_TYPE})...")
if XAIModule is not None:
    try:
        xai = XAIModule(model_weights='yolov8n.pt', conf_threshold=0.25, device=None)
        print(f"✅ XAI initialized in {XAI_TYPE} mode")
    except Exception as e:
        print(f"⚠️  XAI initialization failed: {str(e)[:60]}... Using demo mode")
        XAI_TYPE = "DEMO"
        xai = None
else:
    print("⚠️  XAI module unavailable. Using demo mode.")

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('xai_outputs', exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def image_to_base64(image_path):
    """Convert image file to base64 string for embedding in JSON."""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None


# HTML Frontend
FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fire Detection with Explainable AI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            max-width: 900px;
            width: 100%;
            padding: 40px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .header h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        
        .upload-section {
            margin-bottom: 40px;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #f8f9ff;
        }
        
        .upload-area:hover {
            border-color: #764ba2;
            background: #f0f2ff;
        }
        
        .upload-area input[type="file"] {
            display: none;
        }
        
        .upload-icon {
            font-size: 3em;
            margin-bottom: 10px;
        }
        
        .upload-text {
            color: #666;
            font-size: 1.1em;
        }
        
        #imagePreview {
            margin-top: 20px;
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            display: none;
        }
        
        .results-section {
            margin-top: 40px;
            display: none;
        }
        
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .prediction-fire {
            background: #ffe5e5;
            border-left: 5px solid #e74c3c;
        }
        
        .prediction-safe {
            background: #e5ffe5;
            border-left: 5px solid #27ae60;
        }
        
        .prediction-text {
            font-size: 1.8em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .confidence-bar {
            background: #ddd;
            border-radius: 10px;
            height: 30px;
            margin: 15px 0;
            overflow: hidden;
        }
        
        .confidence-fill {
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 0.3s ease;
        }
        
        .explanation-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .explanation-title {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
        }
        
        .reasoning-list {
            list-style: none;
        }
        
        .reasoning-list li {
            padding: 8px 0;
            color: #555;
            border-bottom: 1px solid #eee;
        }
        
        .reasoning-list li:last-child {
            border-bottom: none;
        }
        
        .visualization-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .viz-item {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }
        
        .viz-item img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .viz-label {
            text-align: center;
            padding: 10px;
            background: #f0f0f0;
            font-weight: bold;
            color: #333;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        
        button {
            flex: 1;
            min-width: 150px;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn-upload {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        
        .btn-upload:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-download {
            background: #f39c12;
            color: white;
        }
        
        .btn-download:hover {
            background: #e67e22;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error-message {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: none;
        }
        
        .success-message {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: none;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 1.8em;
            }
            
            .visualization-gallery {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔥 Fire Detection AI</h1>
            <p>Upload an image to detect fire and get explainable insights</p>
        </div>
        
        <div class="error-message" id="errorMessage"></div>
        <div class="success-message" id="successMessage"></div>
        
        <div class="upload-section">
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📸</div>
                <div class="upload-text">
                    <p>Click to upload or drag and drop</p>
                    <p style="font-size: 0.9em; color: #999;">PNG, JPG, JPEG, BMP up to 50MB</p>
                </div>
                <input type="file" id="fileInput" accept="image/*">
            </div>
            <img id="imagePreview">
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 15px; color: #666;">Processing image...</p>
        </div>
        
        <div class="results-section" id="resultsSection">
            <div class="prediction-box" id="predictionBox">
                <div class="prediction-text" id="predictionText"></div>
                <div>
                    <strong id="confidenceText"></strong>
                    <div class="confidence-bar">
                        <div class="confidence-fill" id="confidenceFill" style="width: 0%;"></div>
                    </div>
                </div>
                <p><strong>Detections:</strong> <span id="detectionsCount"></span></p>
                <p><strong>Image Size:</strong> <span id="imageSize"></span></p>
            </div>
            
            <div class="explanation-section">
                <div class="explanation-title">📊 Reasoning & Analysis</div>
                <ul class="reasoning-list" id="reasoningList"></ul>
            </div>
            
            <div class="visualization-gallery" id="visualizationGallery"></div>
            
            <div class="button-group">
                <button class="btn-upload" onclick="document.getElementById('fileInput').click()">📸 Upload Another</button>
                <button class="btn-download" id="downloadBtn" onclick="downloadResults()">⬇️ Download Results</button>
            </div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const imagePreview = document.getElementById('imagePreview');
        const loading = document.getElementById('loading');
        const resultsSection = document.getElementById('resultsSection');
        const errorMessage = document.getElementById('errorMessage');
        const successMessage = document.getElementById('successMessage');
        
        let currentResult = null;
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#764ba2';
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.borderColor = '#667eea';
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect();
            }
        });
        
        uploadArea.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', handleFileSelect);
        
        function handleFileSelect() {
            const file = fileInput.files[0];
            if (!file) return;
            
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
            
            // Upload and process
            uploadAndProcess(file);
        }
        
        function uploadAndProcess(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            loading.style.display = 'block';
            resultsSection.style.display = 'none';
            errorMessage.style.display = 'none';
            
            fetch('/api/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';
                if (data.error) {
                    showError(data.error);
                } else {
                    currentResult = data;
                    displayResults(data);
                    resultsSection.style.display = 'block';
                    successMessage.style.display = 'block';
                    successMessage.textContent = '✓ Processing complete!';
                    setTimeout(() => { successMessage.style.display = 'none'; }, 3000);
                }
            })
            .catch(error => {
                loading.style.display = 'none';
                showError('Error: ' + error.message);
            });
        }
        
        function displayResults(data) {
            // Prediction box
            const predictionBox = document.getElementById('predictionBox');
            const predictionText = document.getElementById('predictionText');
            const isFire = data.fire_detected;
            
            predictionBox.className = 'prediction-box ' + (isFire ? 'prediction-fire' : 'prediction-safe');
            predictionText.textContent = isFire ? '🔥 FIRE DETECTED' : '✓ NO FIRE';
            
            // Confidence
            const confidence = Math.round(data.confidence * 100);
            document.getElementById('confidenceText').textContent = `Confidence: ${confidence}%`;
            document.getElementById('confidenceFill').style.width = confidence + '%';
            document.getElementById('confidenceFill').textContent = confidence + '%';
            
            document.getElementById('detectionsCount').textContent = data.num_detections;
            document.getElementById('imageSize').textContent = data.image_size;
            
            // Reasoning
            const reasoningList = document.getElementById('reasoningList');
            reasoningList.innerHTML = '';
            data.explanation_details.reasons.forEach(reason => {
                const li = document.createElement('li');
                li.textContent = reason;
                reasoningList.appendChild(li);
            });
            
            # Visualizations
            const gallery = document.getElementById('visualizationGallery');
            gallery.innerHTML = '';
            
            if (data.visualizations) {
                if (data.visualizations.annotated) {
                    addVisualization(gallery, data.visualizations.annotated, 'Bounding Boxes');
                }
                if (data.visualizations.gradcam) {
                    addVisualization(gallery, data.visualizations.gradcam, 'GradCAM Attention');
                }
                if (data.visualizations.explanation) {
                    addVisualization(gallery, data.visualizations.explanation, 'Explanation');
                }
            }
        }
        
        function addVisualization(gallery, base64, label) {
            const item = document.createElement('div');
            item.className = 'viz-item';
            item.innerHTML = `
                <img src="data:image/png;base64,${base64}" alt="${label}">
                <div class="viz-label">${label}</div>
            `;
            gallery.appendChild(item);
        }
        
        function downloadResults() {
            if (!currentResult) return;
            
            // Create JSON download
            const jsonStr = JSON.stringify(currentResult, null, 2);
            const blob = new Blob([jsonStr], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'fire_detection_result.json';
            a.click();
            URL.revokeObjectURL(url);
        }
        
        function showError(message) {
            errorMessage.textContent = '❌ ' + message;
            errorMessage.style.display = 'block';
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Serve the frontend."""
    return render_template_string(FRONTEND_HTML)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Receive image upload, run fire detection, and return explanation.
    Falls back to demo mode if XAI unavailable.
    """
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Use PNG, JPG, JPEG, BMP, or TIFF'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run XAI prediction or use demo mode
        if xai is not None:
            result = xai.explain_prediction(filepath, output_dir='xai_outputs')
        else:
            # Demo mode: generate simulated results
            import random
            confidence = round(random.uniform(0.3, 0.95), 3)
            fire_detected = confidence > 0.5
            result = {
                'prediction': 'FIRE DETECTED' if fire_detected else 'NO FIRE',
                'fire_detected': fire_detected,
                'confidence': confidence,
                'num_detections': random.randint(1, 3) if fire_detected else 0,
                'explanation_text': f"Demo mode: Image analyzed with simulated confidence {confidence:.1%}",
                'explanation_details': {'mode': 'demo', 'note': 'XAI unavailable, showing demo results'},
                'image_size': (640, 480),
                'output_files': {}
            }
        
        # Convert images to base64 for embedding in JSON response
        visualizations = {}
        for key in ['annotated', 'heatmap', 'explanation']:
            if key in result.get('output_files', {}):
                base64_img = image_to_base64(result['output_files'][key])
                if base64_img:
                    visualizations[key] = base64_img
        
        # Build response
        response = {
            'prediction': result['prediction'],
            'fire_detected': result['fire_detected'],
            'confidence': result['confidence'],
            'num_detections': result.get('num_detections', 0),
            'explanation_text': result.get('explanation_text', 'N/A'),
            'explanation_details': result.get('explanation_details', {}),
            'image_size': result.get('image_size', (0, 0)),
            'visualizations': visualizations,
            'timestamp': str(os.path.getmtime(filepath)),
            'xai_mode': XAI_TYPE
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok', 
        'model': 'yolov8n', 
        'xai': 'enabled' if xai is not None else 'demo_mode',
        'xai_type': XAI_TYPE
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔥 Fire Detection API with Explainable AI (GradCAM)")
    print("="*60)
    print(f"\n✓ XAI Type: {XAI_TYPE}")
    print("✓ Starting Flask server...")
    print("✓ Frontend: http://localhost:5000")
    print("✓ API: http://localhost:5000/api/predict")
    print("✓ Health: http://localhost:5000/api/health")
    print("\n" + "="*60 + "\n")
    
    # Disable reloader to fix Windows signal issue
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
