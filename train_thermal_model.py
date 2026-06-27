#!/usr/bin/env python3
"""
Train YOLOv8 model on thermal fire detection dataset (5 epochs for quick accurate training).

This script:
1. Prepares the YOLO dataset from thermal_detection_results/annotated_images
2. Fine-tunes YOLOv8 model for 5 epochs
3. Saves the best trained model for use with GradCAM XAI

Usage:
  python train_thermal_model.py --images_dir thermal_detection_results/annotated_images \
    --labels_dir thermal_detection_results/labels --epochs 5 --batch 16 --model yolov8n.pt

Output:
  - Trained model: runs/train/thermal_fire/weights/best.pt
  - Use this with GradCAM: python run_gradcam_single_image.py --weights runs/train/thermal_fire/weights/best.pt
"""

import argparse
import sys
from pathlib import Path
import shutil
from prepare_yolo_dataset import prepare_dataset

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 on thermal fire detection dataset (5 epochs)')
    parser.add_argument('--images_dir', required=True, help='Path to annotated images folder')
    parser.add_argument('--labels_dir', required=True, help='Path to labels folder')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs (default: 5)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--model', default='yolov8n.pt', help='Base YOLOv8 model (default: yolov8n.pt)')
    parser.add_argument('--device', default=None, help='Device (cuda, cpu, or None for auto)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size (default: 640)')
    parser.add_argument('--project', default='runs/train', help='Project directory for outputs')
    parser.add_argument('--name', default='thermal_fire', help='Experiment name')
    parser.add_argument('--dataset_out', default='yolo_thermal_dataset', help='Temporary dataset folder')
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    dataset_out = Path(args.dataset_out)

    # Validate inputs
    if not images_dir.exists():
        print(f"❌ Error: Images directory not found: {images_dir}")
        sys.exit(1)
    if not labels_dir.exists():
        print(f"❌ Error: Labels directory not found: {labels_dir}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("🔥 THERMAL FIRE DETECTION - YOLOv8 TRAINING PIPELINE")
    print("=" * 80)

    # Step 1: Prepare dataset
    print("\n📊 STEP 1: Preparing YOLO Dataset")
    print("-" * 80)
    try:
        prepare_dataset(
            images_dir=images_dir,
            labels_dir=labels_dir,
            out_dir=dataset_out,
            split=(0.8, 0.1, 0.1),  # 80% train, 10% val, 10% test
            force_full_bbox=True,
            seed=42
        )
        print("✅ Dataset prepared successfully")
    except Exception as e:
        print(f"❌ Error preparing dataset: {e}")
        sys.exit(1)

    # Step 2: Train model
    print("\n🚀 STEP 2: Training YOLOv8 Model")
    print("-" * 80)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch}")
    print(f"Image Size: {args.imgsz}x{args.imgsz}")
    print(f"Device: {args.device or 'auto (GPU if available, else CPU)'}")
    print(f"Project: {args.project}")
    print(f"Name: {args.name}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n❌ Error: ultralytics not installed")
        print("Install it with: pip install -r requirements_xai.txt")
        sys.exit(1)

    try:
        # Load model
        print(f"\n⏳ Loading model: {args.model}")
        model = YOLO(args.model)

        # Prepare data.yaml path
        data_yaml = dataset_out / 'data.yaml'
        if not data_yaml.exists():
            print(f"❌ Error: data.yaml not found at {data_yaml}")
            sys.exit(1)

        # Train
        print(f"⏳ Training for {args.epochs} epochs...")
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            patience=3,  # Early stopping patience
            save=True,
            verbose=True,
            plots=True,
        )

        # Get best weights path
        best_weights = Path(args.project) / args.name / 'weights' / 'best.pt'

        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\n📊 Training Results:")
        print(f"  • Model saved: {best_weights}")
        print(f"  • Last model: {Path(args.project) / args.name / 'weights' / 'last.pt'}")
        print(f"  • Results log: {Path(args.project) / args.name / 'results.csv'}")

        # Display metrics if available
        if results and hasattr(results, 'results_dict'):
            print(f"\n📈 Final Metrics:")
            metrics = results.results_dict
            if 'metrics/precision(B)' in metrics:
                print(f"  • Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
            if 'metrics/recall(B)' in metrics:
                print(f"  • Recall: {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
            if 'metrics/mAP50(B)' in metrics:
                print(f"  • mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
            if 'metrics/mAP50-95(B)' in metrics:
                print(f"  • mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")

        print(f"\n🎯 NEXT STEPS:")
        print(f"  1. Use trained model with GradCAM:")
        print(f"     python select_and_run_gradcam.py --weights '{best_weights}'")
        print(f"\n  2. Or run single image analysis:")
        print(f"     python run_gradcam_single_image.py --image 'thermal_detection_results/annotated_images/image.png' --weights '{best_weights}'")
        print(f"\n  3. View training results:")
        print(f"     • Plots: {Path(args.project) / args.name}")
        print(f"     • CSV Log: {Path(args.project) / args.name / 'results.csv'}")

        print("\n" + "=" * 80)
        print(f"✨ Model ready for fire detection with GradCAM XAI! ✨")
        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Optional: Clean up temporary dataset (comment out if you want to keep it for inspection)
        # if dataset_out.exists():
        #     shutil.rmtree(dataset_out)
        pass


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
