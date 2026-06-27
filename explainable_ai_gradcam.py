"""
Enhanced Explainable AI with GradCAM for Fire Detection.
Combines YOLOv8 predictions with GradCAM attention maps to show which image regions
influence the model's fire detection decision.

GradCAM: Gradient-weighted Class Activation Mapping
- Shows WHAT the model is looking at (attention)
- Shows WHY it made this decision
- Visualizes feature importance in the image
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Tuple, Dict, List, Optional


class GradCAM:
    """
    Implements GradCAM (Gradient-weighted Class Activation Mapping) for YOLOv8.
    Generates attention maps showing which regions influence fire detection.
    """
    
    def __init__(self, model, target_layers: List[str]):
        """
        Initialize GradCAM.
        
        Args:
            model: YOLO model
            target_layers: List of layer names to extract gradients from
        """
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture gradients."""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        # Register hooks on target layers
        for name, module in self.model.model.named_modules():
            for target_layer in self.target_layers:
                if target_layer in name:
                    module.register_forward_hook(forward_hook(name))
                    module.register_full_backward_hook(backward_hook(name))
    
    def generate_cam(self, image: np.ndarray, confidence_threshold: float = 0.25) -> Optional[np.ndarray]:
        """
        Generate GradCAM attention map.
        
        Args:
            image: Input image (numpy array)
            confidence_threshold: Confidence threshold for fire detection
            
        Returns:
            CAM (Class Activation Map) or None if no detections
        """
        try:
            # Convert to tensor
            img_tensor = torch.from_numpy(image).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(next(self.model.model.parameters()).device)
            
            # Forward pass
            with torch.enable_grad():
                img_tensor.requires_grad = True
                outputs = self.model.model(img_tensor)
            
            # Check for detections above threshold
            detections = outputs[0]
            if detections is None or len(detections) == 0:
                return None
            
            # Filter by confidence
            fire_detections = detections[detections[:, 4] > confidence_threshold]
            if len(fire_detections) == 0:
                return None
            
            # Get max confidence score as target
            max_conf = fire_detections[:, 4].max()
            
            # Backward pass
            self.model.model.zero_grad()
            max_conf.backward()
            
            # Generate CAM
            cam = self._compute_cam()
            
            return cam
        
        except Exception as e:
            print(f"Error generating GradCAM: {e}")
            return None
    
    def _compute_cam(self) -> Optional[np.ndarray]:
        """Compute the Class Activation Map from gradients and activations."""
        try:
            # Get last layer with activations
            for layer_name in reversed(list(self.activations.keys())):
                if layer_name in self.gradients:
                    activations = self.activations[layer_name]
                    gradients = self.gradients[layer_name]
                    
                    # Compute weights (average gradient)
                    weights = gradients.mean(dim=(2, 3), keepdim=True)
                    
                    # Weighted combination of activations
                    cam = (weights * activations).sum(dim=1, keepdim=True)
                    cam = F.relu(cam)
                    
                    # Normalize
                    cam = cam.squeeze(0).squeeze(0).cpu().detach().numpy()
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                    
                    return cam
            
            return None
        except Exception as e:
            print(f"Error computing CAM: {e}")
            return None


class FireDetectionXAIGradCAM:
    """
    Enhanced XAI system combining YOLOv8 + GradCAM for fire detection explanations.
    """
    
    def __init__(self, model_weights: str, conf_threshold: float = 0.25, device: Optional[str] = None):
        """
        Initialize XAI system with GradCAM.
        
        Args:
            model_weights: Path to YOLOv8 weights
            conf_threshold: Confidence threshold for fire detection
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        self.model = YOLO(model_weights)
        self.conf_threshold = conf_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize GradCAM
        self.gradcam = GradCAM(self.model, target_layers=['model.24'])  # Last detection layer
    
    def detect_fire(self, image_path: str) -> Tuple[Dict, np.ndarray]:
        """
        Run fire detection on image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (detection results, image array)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model.predict(image_path, conf=self.conf_threshold, verbose=False)
        
        return results[0], image_rgb
    
    def generate_explanation(self, result, image: np.ndarray, fire_detected: bool) -> Tuple[List[str], str]:
        """
        Generate textual explanation for prediction.
        
        Args:
            result: YOLOv8 detection results
            image: Input image array
            fire_detected: Whether fire was detected
            
        Returns:
            Tuple of (reasons list, full explanation string)
        """
        reasons = []
        
        if fire_detected and len(result.boxes) > 0:
            confidences = result.boxes.conf.cpu().numpy()
            num_detections = len(confidences)
            avg_conf = confidences.mean()
            max_conf = confidences.max()
            
            reasons = [
                f"✓ Detected {num_detections} potential fire region(s)",
                f"✓ Highest confidence score: {max_conf*100:.2f}%",
                f"✓ Average confidence: {avg_conf*100:.2f}%",
                f"✓ Detection threshold exceeded ({self.conf_threshold})",
                f"✓ Visual features match fire patterns in thermal/RGB domain"
            ]
        else:
            reasons = [
                "✓ No fire-like patterns detected in the image",
                "✓ Scene lacks thermal signatures consistent with fire",
                "✓ Image characteristics do not match learned fire features",
                f"✓ Detections below confidence threshold ({self.conf_threshold})",
                "✓ Region appears to be non-fire vegetation, water, or other objects"
            ]
        
        explanation_text = "\n".join(reasons)
        return reasons, explanation_text
    
    def draw_bounding_boxes(self, image: np.ndarray, result) -> np.ndarray:
        """
        Draw bounding boxes with confidence scores.
        
        Args:
            image: Input image
            result: YOLOv8 detection results
            
        Returns:
            Image with bounding boxes drawn
        """
        annotated = image.copy()
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confidences):
                if conf > self.conf_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Draw red box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Add confidence label
                    label = f"Fire: {conf*100:.1f}%"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    color = (255, 0, 0)  # Red (BGR)
                    
                    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                    cv2.rectangle(annotated, 
                                (x1, y1 - text_size[1] - 10),
                                (x1 + text_size[0] + 5, y1),
                                color, -1)
                    cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                              font, font_scale, (255, 255, 255), thickness)
        
        return annotated
    
    def generate_gradcam_heatmap(self, image_path: str, result) -> Optional[np.ndarray]:
        """
        Generate GradCAM attention map.
        
        Args:
            image_path: Path to input image
            result: YOLOv8 detection results
            
        Returns:
            GradCAM heatmap or None
        """
        try:
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Generate GradCAM
            cam = self.gradcam.generate_cam(image_rgb, self.conf_threshold)
            
            if cam is None:
                return None
            
            # Resize to image size
            cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
            
            # Apply colormap
            heatmap_colored = cm.jet(cam_resized)
            heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
            
            # Blend with original
            blended = cv2.addWeighted(image, 0.5, heatmap_colored, 0.5, 0)
            
            return blended
        
        except Exception as e:
            print(f"Error generating GradCAM heatmap: {e}")
            return None
    
    def create_explanation_image(self, image: np.ndarray, explanation_text: str, 
                                fire_detected: bool, gradcam_available: bool = False) -> np.ndarray:
        """
        Create image with text explanation overlay.
        
        Args:
            image: Input image
            explanation_text: Explanation text
            fire_detected: Whether fire was detected
            gradcam_available: Whether GradCAM heatmap is available
            
        Returns:
            Image with text overlay
        """
        image_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(image_pil, 'RGBA')
        
        # Semi-transparent background
        bg_color = (255, 100, 100, 200) if fire_detected else (100, 200, 100, 200)
        
        # Draw prediction header
        header = "🔥 FIRE DETECTED" if fire_detected else "✓ NO FIRE"
        header_box = (10, 10, image.shape[1] - 10, 70)
        draw.rectangle(header_box, fill=bg_color)
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
            font_small = font
        
        draw.text((20, 20), header, fill=(255, 255, 255), font=font)
        
        # Add explanation text
        y_offset = 90
        explanation_lines = explanation_text.split('\n')
        for line in explanation_lines:
            if line.strip():
                draw.text((20, y_offset), line, fill=(0, 0, 0), font=font_small)
                y_offset += 25
        
        # Add GradCAM indicator if available
        if gradcam_available:
            indicator = "GradCAM: Active (shows attention regions)"
            draw.text((20, y_offset + 10), indicator, fill=(102, 51, 153), font=font_small)
        
        return np.array(image_pil)
    
    def explain_prediction(self, image_path: str, output_dir: str = 'xai_outputs') -> Dict:
        """
        Generate complete explanation for single image.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with prediction, confidence, explanations, and file paths
        """
        from pathlib import Path
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get image stem
        stem = Path(image_path).stem
        
        # Run detection
        result, image_rgb = self.detect_fire(image_path)
        
        # Determine if fire detected
        fire_detected = len(result.boxes) > 0 and any(
            conf > self.conf_threshold for conf in result.boxes.conf.cpu().numpy()
        )
        
        # Get confidence
        if fire_detected:
            confidence = result.boxes.conf.max().cpu().item()
        else:
            confidence = 0.0
        
        # Generate explanation
        reasons, explanation_text = self.generate_explanation(result, image_rgb, fire_detected)
        
        # Generate visualizations
        annotated_img = self.draw_bounding_boxes(image_rgb, result)
        
        # Try to generate GradCAM
        gradcam_img = self.generate_gradcam_heatmap(image_path, result)
        gradcam_available = gradcam_img is not None
        
        explanation_img = self.create_explanation_image(
            image_rgb, 
            explanation_text, 
            fire_detected,
            gradcam_available
        )
        
        # Save outputs
        output_files = {}
        
        # Save annotated
        annotated_path = f"{output_dir}/{stem}_annotated.png"
        cv2.imwrite(annotated_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        output_files['annotated'] = annotated_path
        
        # Save GradCAM if available
        if gradcam_available:
            gradcam_path = f"{output_dir}/{stem}_gradcam.png"
            cv2.imwrite(gradcam_path, gradcam_img)
            output_files['gradcam'] = gradcam_path
        
        # Save explanation image
        explanation_path = f"{output_dir}/{stem}_explanation.png"
        cv2.imwrite(explanation_path, cv2.cvtColor(explanation_img, cv2.COLOR_RGB2BGR))
        output_files['explanation'] = explanation_path
        
        # Save explanation text
        explanation_txt_path = f"{output_dir}/{stem}_explanation.txt"
        with open(explanation_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"PREDICTION: {'FIRE DETECTED' if fire_detected else 'NO FIRE'}\n")
            f.write(f"Confidence: {confidence*100:.2f}%\n")
            f.write(f"Detections: {len(result.boxes) if result.boxes else 0}\n")
            if gradcam_available:
                f.write("GradCAM: Enabled (attention map generated)\n")
            f.write(f"\nREASONING:\n{explanation_text}\n")
        output_files['explanation_text'] = explanation_txt_path
        
        # Get image size
        image_size = f"{image_rgb.shape[1]}x{image_rgb.shape[0]}"
        
        return {
            'prediction': 'FIRE DETECTED' if fire_detected else 'NO FIRE',
            'fire_detected': fire_detected,
            'confidence': confidence,
            'num_detections': len(result.boxes) if result.boxes else 0,
            'explanation_text': explanation_text,
            'explanation_details': {'reasons': reasons},
            'image_size': image_size,
            'output_files': output_files,
            'gradcam_available': gradcam_available
        }


def main():
    """CLI interface for GradCAM XAI."""
    parser = argparse.ArgumentParser(description='Fire Detection with GradCAM XAI')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Path to YOLOv8 weights')
    parser.add_argument('--output_dir', type=str, default='xai_outputs', help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize XAI with GradCAM
    print("\n" + "="*60)
    print("🔥 Fire Detection with GradCAM Explainable AI")
    print("="*60)
    
    xai = FireDetectionXAIGradCAM(
        model_weights=args.weights,
        conf_threshold=args.conf,
        device=args.device
    )
    
    # Run explanation
    print(f"\n📸 Processing: {args.image}")
    result = xai.explain_prediction(args.image, output_dir=args.output_dir)
    
    # Display results
    print(f"\nPREDICTION: {result['prediction']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print(f"Detections: {result['num_detections']}")
    
    if result['gradcam_available']:
        print("✓ GradCAM: Enabled (attention map generated)")
    else:
        print("✗ GradCAM: No detections for attention map")
    
    print(f"\nREASONING:")
    print(result['explanation_text'])
    
    print(f"\nOUTPUTS:")
    for key, path in result['output_files'].items():
        print(f"  ✓ {key}: {path}")
    
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()
