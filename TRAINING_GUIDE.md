# 🔥 Training YOLOv8 on Thermal Fire Detection Data

## Quick Start (3 Steps)

### Step 1: Prepare Dataset
Organize your thermal images and labels into YOLO format:

```powershell
python .\prepare_yolo_dataset.py `
  --images_dir thermal_detection_results/annotated_images `
  --labels_dir thermal_detection_results/labels `
  --out yolo_thermal_dataset `
  --force_full_bbox
```

**What this does:**
- Creates `train/`, `valid/`, `test/` folders with images and labels
- Splits data: 80% train, 10% validation, 10% test
- Converts labels to YOLO format (class_id center_x center_y width height)
- Generates `data.yaml` for training

**Output:** `yolo_thermal_dataset/` folder with YOLO-format data

### Step 2: Train Model (5 Epochs)
Fine-tune YOLOv8 with just 5 epochs for accurate results:

```powershell
python .\train_thermal_model.py `
  --images_dir thermal_detection_results/annotated_images `
  --labels_dir thermal_detection_results/labels `
  --epochs 5 `
  --batch 16 `
  --model yolov8n.pt
```

**What this does:**
- Loads pre-trained YOLOv8 nano model
- Trains on thermal fire detection data for 5 epochs
- Saves best model to `runs/train/thermal_fire/weights/best.pt`
- Generates training plots and metrics

**Training time:**
- GPU (CUDA): ~2-5 minutes
- CPU: ~15-30 minutes

**Output:** `runs/train/thermal_fire/`

### Step 3: Run GradCAM XAI with Trained Model
Use the newly trained model for explainable fire detection:

```powershell
# Interactive image selection
python .\select_and_run_gradcam.py --weights runs/train/thermal_fire/weights/best.pt

# Or specify image directly
python .\run_gradcam_single_image.py `
  --image thermal_detection_results/annotated_images/image.png `
  --weights runs/train/thermal_fire/weights/best.pt
```

---

## Detailed Commands

### Option A: Full Pipeline (Auto)
Run everything in one command:

```powershell
python .\train_thermal_model.py `
  --images_dir thermal_detection_results/annotated_images `
  --labels_dir thermal_detection_results/labels `
  --epochs 5 `
  --batch 16 `
  --model yolov8n.pt `
  --device 0
```

**Arguments explained:**
- `--epochs 5` : Train for exactly 5 epochs (quick, accurate)
- `--batch 16` : 16 images per batch (adjust if GPU memory is limited, e.g., 8)
- `--model yolov8n.pt` : Nano model (fast; try `yolov8s.pt` for higher accuracy)
- `--device 0` : Use GPU device 0 (use `cpu` for CPU-only)

### Option B: Manual Steps

**1. Prepare dataset:**
```powershell
python .\prepare_yolo_dataset.py `
  --images_dir thermal_detection_results/annotated_images `
  --labels_dir thermal_detection_results/labels `
  --out my_dataset `
  --force_full_bbox
```

**2. Train with custom settings:**
```powershell
python .\pccoe\thermal_forestfire_detection.v2i.yolov8-obb\train.py `
  --model yolov8n.pt `
  --data my_dataset/data.yaml `
  --epochs 5 `
  --batch 16 `
  --imgsz 640 `
  --device 0
```

**3. Use trained model:**
```powershell
python .\run_gradcam_single_image.py `
  --image thermal_detection_results/annotated_images/image.png `
  --weights runs/train/exp/weights/best.pt `
  --conf 0.25 `
  --output_dir my_results
```

---

## Training Parameters Explained

| Parameter | Default | Recommendation | Notes |
|-----------|---------|-----------------|-------|
| `--epochs` | 5 | 5 | Quick training; increase to 10-20 for better accuracy |
| `--batch` | 16 | 8-16 | Larger = faster (if GPU memory allows); smaller = more stable |
| `--imgsz` | 640 | 640 | Image size; smaller (416) = faster, larger (1280) = more accurate |
| `--model` | yolov8n.pt | yolov8n.pt | nano=fast, small=balanced, medium=accurate |
| `--device` | auto | 0 (GPU) or cpu | Use GPU if available for 10x speedup |

---

## Output Files

After training at `runs/train/thermal_fire/`:

```
runs/train/thermal_fire/
├── weights/
│   ├── best.pt          ← Use this with GradCAM
│   └── last.pt
├── results.csv          ← Training metrics
├── confusion_matrix.png ← Class confusion
└── plots/
    ├── precision_recall.png
    ├── f1_curve.png
    ├── training_loss.png
    └── val_metrics.png
```

### Interpreting Metrics

**Check `results.csv`:**
```
epoch, train_loss, val_loss, precision, recall, mAP50, mAP50-95
...
4,     0.245,     0.312,    0.92,      0.88,   0.87,  0.65
```

- **Precision**: Of detected fires, how many are real? (higher = fewer false alarms)
- **Recall**: Of actual fires, how many did we detect? (higher = fewer missed fires)
- **mAP50**: Average precision at IoU=0.5 (detection accuracy)
- **mAP50-95**: Average precision at IoU=0.5:0.95 (stricter metric)

---

## Troubleshooting

### ❌ "ultralytics not installed"
```powershell
pip install ultralytics torch torchvision opencv-python pillow
```

### ❌ "CUDA out of memory"
- Reduce `--batch` size: `--batch 8`
- Reduce `--imgsz`: `--imgsz 416`
- Use CPU: `--device cpu`

### ❌ "No images found"
- Check folder path exists: `ls thermal_detection_results/annotated_images`
- Ensure images have `.png` or `.jpg` extension
- Count: `(ls thermal_detection_results/annotated_images -Filter *.png | Measure-Object).Count`

### ❌ "No labels found"
- Check folder path: `ls thermal_detection_results/labels`
- Ensure label files have `.txt` extension
- Count: `(ls thermal_detection_results/labels -Filter *.txt | Measure-Object).Count`

### ❌ Training is slow
- Use GPU: `--device 0` (check with `nvidia-smi`)
- Reduce image size: `--imgsz 416`
- Reduce batch size: `--batch 8` (trains longer but may generalize better)

### ✅ Training succeeded but low accuracy
- Use more epochs: `--epochs 10` (or more)
- Better model: `--model yolov8s.pt` (small model)
- Check data quality: Are labels correctly annotated?
- Try higher resolution: `--imgsz 1280`

---

## Advanced Usage

### Train with custom hyperparameters:
```powershell
python .\train_thermal_model.py `
  --images_dir thermal_detection_results/annotated_images `
  --labels_dir thermal_detection_results/labels `
  --epochs 10 `
  --batch 8 `
  --model yolov8s.pt `
  --imgsz 1280 `
  --device 0 `
  --project my_experiments `
  --name thermal_v2
```

### Resume training from checkpoint:
```powershell
python .\pccoe\thermal_forestfire_detection.v2i.yolov8-obb\train.py `
  --model runs/train/thermal_fire/weights/last.pt `
  --data yolo_thermal_dataset/data.yaml `
  --epochs 10 `
  --batch 16
```

### Validate/test trained model:
```powershell
python .\pccoe\thermal_forestfire_detection.v2i.yolov8-obb\eval.py `
  --model runs/train/thermal_fire/weights/best.pt `
  --data yolo_thermal_dataset/data.yaml
```

---

## GradCAM Integration

Once training is complete, use the model with GradCAM for explainable predictions:

### 1. Interactive mode (GUI file picker):
```powershell
python .\select_and_run_gradcam.py --weights runs/train/thermal_fire/weights/best.pt
```

### 2. Batch analysis (all images in folder):
```powershell
python .\batch_gradcam_analysis.py `
  --images_dir thermal_detection_results/annotated_images `
  --weights runs/train/thermal_fire/weights/best.pt `
  --output_dir thermal_gradcam_results
```

### 3. Single image with full options:
```powershell
python .\run_gradcam_single_image.py `
  --image thermal_detection_results/annotated_images/image.png `
  --weights runs/train/thermal_fire/weights/best.pt `
  --conf 0.25 `
  --output_dir my_results
```

---

## Performance Tips

### For Faster Training:
- ✅ Use GPU (`--device 0`)
- ✅ Reduce batch size to fit in memory
- ✅ Use nano model (`yolov8n.pt`)
- ✅ Reduce image size (`--imgsz 416`)
- ❌ Avoid CPU training (very slow)

### For Better Accuracy:
- ✅ Train longer (`--epochs 20` or more)
- ✅ Use larger model (`yolov8s.pt` or `yolov8m.pt`)
- ✅ Increase image size (`--imgsz 1280`)
- ✅ Ensure high-quality, well-annotated data
- ✅ Use data augmentation (automatic with YOLO)

### For Production:
- ✅ Use `runs/train/thermal_fire/weights/best.pt` (not `last.pt`)
- ✅ Monitor validation metrics in `results.csv`
- ✅ Stop early if validation loss plateaus
- ✅ Save and version your trained models

---

## Next Steps

1. **Run training:**
   ```powershell
   python .\train_thermal_model.py --images_dir thermal_detection_results/annotated_images --labels_dir thermal_detection_results/labels --epochs 5
   ```

2. **Check results:**
   ```powershell
   explorer runs\train\thermal_fire
   ```

3. **Use with GradCAM:**
   ```powershell
   python .\select_and_run_gradcam.py --weights runs/train/thermal_fire/weights/best.pt
   ```

4. **View explanations:**
   - Open `xai_outputs/` folder
   - View `*_gradcam.png` (attention heatmap)
   - View `*_explanation.txt` (reasoning)

---

## Summary

| Step | Command | Time |
|------|---------|------|
| 1. Prepare | `python prepare_yolo_dataset.py ...` | 1 min |
| 2. Train | `python train_thermal_model.py --epochs 5` | 2-5 min (GPU) / 15-30 min (CPU) |
| 3. Analyze | `python select_and_run_gradcam.py --weights best.pt` | 1-5 sec per image |

**Total: 5-10 minutes (GPU) or 20-40 minutes (CPU) to go from raw images to trained model with XAI!**

---

## Questions?

- Check `QUICK_REFERENCE.md` for GradCAM overview
- Read `GRADCAM_GUIDE.md` for technical details
- See `XAI_README.md` for full setup instructions
- Review `ARCHITECTURE_DIAGRAMS.md` for system overview

**Let's train and detect fires with full explainability! 🔥**
