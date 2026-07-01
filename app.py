"""
Flask API for Fire Detection with Explainable AI.
Allows users to upload images and receive fire predictions with visual explanations.
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
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
    print("âœ… GradCAM XAI module loaded successfully")
except Exception as e:
    print(f"âš ï¸  GradCAM import failed: {str(e)[:60]}...")
    try:
        from explainable_ai import FireDetectionXAI as XAIModule
        XAI_TYPE = "Standard"
        print("âœ… Standard XAI module loaded successfully")
    except Exception as e2:
        print(f"âš ï¸  Standard XAI also failed: {str(e2)[:60]}...")
        print("ðŸŽ¬ Running in DEMO MODE (simulated results)")
        XAI_TYPE = "DEMO"
        XAIModule = None

app = Flask(__name__, static_folder='frontend', static_url_path='')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Path to frontend folder
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend')

# Initialize XAI module only if available
print(f"Initializing XAI module ({XAI_TYPE})...")
if XAIModule is not None:
    try:
        xai = XAIModule(model_weights='yolov8n.pt', conf_threshold=0.25, device=None)
        print(f"âœ… XAI initialized in {XAI_TYPE} mode")
    except Exception as e:
        print(f"âš ï¸  XAI initialization failed: {str(e)[:60]}... Using demo mode")
        XAI_TYPE = "DEMO"
        xai = None
else:
    print("âš ï¸  XAI module unavailable. Using demo mode.")

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



# â”€â”€ Frontend File Routes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route('/')
def index():
    """Serve the main frontend page."""
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/style.css')
def stylesheet():
    """Serve the CSS stylesheet."""
    return send_from_directory(FRONTEND_DIR, 'style.css')


@app.route('/script.js')
def javascript():
    """Serve the JavaScript file."""
    return send_from_directory(FRONTEND_DIR, 'script.js')



@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Receive image upload, run fire detection, and return explanation.
    Falls back to demo mode if XAI unavailable.
    """
    try:
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
            import random
            confidence = round(random.uniform(0.3, 0.95), 3)
            fire_detected = confidence > 0.5
            result = {
                'prediction': 'FIRE DETECTED' if fire_detected else 'NO FIRE',
                'fire_detected': fire_detected,
                'confidence': confidence,
                'num_detections': random.randint(1, 3) if fire_detected else 0,
                'explanation_text': f"Demo mode: Image analyzed with simulated confidence {confidence:.1%}",
                'explanation_details': {
                    'mode': 'demo',
                    'note': 'XAI unavailable, showing demo results',
                    'reasons': [
                        f"Simulated confidence score: {confidence:.1%}",
                        f"{'Fire signatures detected in thermal range' if fire_detected else 'No thermal anomalies detected'}",
                        "Running in DEMO mode â€” attach real model for live results",
                        f"Detections: {random.randint(1,3) if fire_detected else 0} region(s) flagged"
                    ]
                },
                'image_size': (640, 480),
                'output_files': {}
            }

        # Convert images to base64
        visualizations = {}
        for key in ['annotated', 'heatmap', 'explanation']:
            if key in result.get('output_files', {}):
                b64 = image_to_base64(result['output_files'][key])
                if b64:
                    visualizations[key] = b64

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
    print("ðŸ”¥ ThermoVision AI â€” Fire Detection (GradCAM)")
    print("="*60)
    print(f"\nâœ“ XAI Type: {XAI_TYPE}")
    print("âœ“ Starting Flask server...")
    print("âœ“ Frontend: http://localhost:5000")
    print("âœ“ API: http://localhost:5000/api/predict")
    print("âœ“ Health: http://localhost:5000/api/health")
    print("\n" + "="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
