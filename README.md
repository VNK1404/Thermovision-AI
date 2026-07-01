# 🔥 Thermovision-AI

**Thermovision‑AI** is a lightweight prototype for detecting wildfire and thermal anomalies from infrared imagery using YOLOv8, CycleGAN domain adaptation, and Grad‑CAM explainability.

---

## ✨ Key Features
- Real‑time fire detection with YOLOv8 fine‑tuned on thermal data.
- Grad‑CAM visual explanations of model decisions.
- Optional CycleGAN translation (RGB → Thermal) for data augmentation.
- Advanced 3D glassmorphism dashboard UI with interactive Grad-CAM slider.


---

## 🚀 Quick Start
### Prerequisites
- Python 3.8+ (recommended via virtual‑env)
- `pip` (or `conda`)
- (Optional) NVIDIA GPU with CUDA for faster training/inference.

### Installation
```bash
# Clone the repo
git clone https://github.com/your-org/Thermovision-AI.git
cd Thermovision-AI

# Create and activate a virtual environment (Windows example)
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements_xai.txt
```

### Run the Pipeline (3 commands)
```bash
# 1️⃣ Prepare YOLO dataset from your thermal images
python prepare_yolo_dataset.py --images_dir thermal_detection_results/annotated_images \
    --labels_dir thermal_detection_results/labels --out yolo_thermal_dataset --force_full_bbox

# 2️⃣ Train the YOLO model (adjust epochs/batch as needed)
python train_thermal_model.py --images_dir thermal_detection_results/annotated_images \
    --labels_dir thermal_detection_results/labels --epochs 5 --batch 16

# 3️⃣ Generate Grad‑CAM explanations for a test image
python explainable_ai_gradcam.py --image path/to/image.png \
    --weights runs/train/thermal_fire/weights/best.pt --output_dir xai_outputs
```

### Launch the Web App
```bash
python app.py
```
Open `http://localhost:5000` in a browser and upload a thermal image to see predictions with Grad‑CAM overlays.

---

## 📡 API Endpoints (Flask)
| Route | Method | Description |
|------|--------|-------------|
| `/` | GET | Interactive UI |
| `/api/predict` | POST | Upload image → returns detection + Grad‑CAM (JSON) |
| `/api/health` | GET | Service health check |

---

## 📦 Dependencies
```text
ultralytics >=8.0.0
torch >=2.0.0 (CUDA optional)
opencv-python >=4.8.0
flask >=3.0.0
numpy >=1.24.0
```
See `requirements_xai.txt` for the full list.

---

## 📄 License
This project is licensed under the MIT License.
