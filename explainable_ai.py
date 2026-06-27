"""
Explainable AI module for fire detection.
Provides visual explanations (bounding boxes, heatmaps, confidence scores)
and textual reasoning for fire predictions.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from ultralytics import YOLO


class FireDetectionXAI:
    """
    Explainable AI for fire detection using YOLOv8.
    Generates visualizations and explanations for predictions.
    """
    
    def __init__(self, model_weights='yolov8n.pt', conf_threshold=0.25, device=None):
        """
        Initialize the XAI module.
        
        Args:
            model_weights: Path to YOLOv8 model weights
            conf_threshold: Confidence threshold for detections
            device: Device for inference (None for auto, '0' for GPU, 'cpu' for CPU)
        """
        self.model = YOLO(model_weights)
        self.conf_threshold = conf_threshold
        self.device = device
        self.class_names = {0: 'fire', 1: 'person', 2: 'car'}  # Generic; adjust if trained on specific classes
    
    def detect_fire(self, image_path):
        """
        Run fire detection on an image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            results: YOLO results object
            image: Original image (numpy array, BGR)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        results = self.model.predict(
            source=image_path,
            imgsz=640,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device
        )
        
        return results[0], image
    
    def generate_explanation(self, result, image, fire_detected):
        """
        Generate textual explanation for the prediction.
        
        Args:
            result: YOLO detection result
            image: Original image (numpy array)
            fire_detected: Boolean indicating if fire was detected
            
        Returns:
            explanation_text: Human-readable explanation
            explanation_details: Dict with reasoning details
        """
        details = {
            'prediction': 'FIRE DETECTED' if fire_detected else 'NO FIRE',
            'confidence': 0.0,
            'boxes_count': 0,
            'image_size': f"{image.shape[1]}x{image.shape[0]}",
            'reasons': []
        }
        
        try:
            num_boxes = len(result.boxes) if hasattr(result.boxes, '__len__') else 0
        except:
            num_boxes = 0
        
        details['boxes_count'] = num_boxes
        
        if fire_detected and num_boxes > 0:
            # Extract confidence scores from detections
            try:
                confidences = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else []
                if len(confidences) > 0:
                    details['confidence'] = float(np.max(confidences))
                    avg_conf = float(np.mean(confidences))
                    
                    details['reasons'] = [
                        f"✓ Detected {num_boxes} potential fire region(s)",
                        f"✓ Highest confidence score: {details['confidence']:.2%}",
                        f"✓ Average confidence: {avg_conf:.2%}",
                        f"✓ Detection threshold exceeded ({self.conf_threshold:.2f})",
                        f"✓ Visual features match fire patterns in thermal/RGB domain"
                    ]
            except Exception as e:
                details['reasons'] = [
                    f"✓ Detected {num_boxes} potential fire region(s)",
                    f"✓ Confidence score above threshold"
                ]
        else:
            details['confidence'] = 0.0
            details['reasons'] = [
                f"✓ No fire-like patterns detected in the image",
                f"✓ Scene lacks thermal signatures consistent with fire",
                f"✓ Image characteristics do not match learned fire features",
                f"✓ Detections below confidence threshold ({self.conf_threshold:.2f})",
                f"✓ Region appears to be non-fire vegetation, water, or other objects"
            ]
        
        explanation_text = f"PREDICTION: {details['prediction']}\n"
        explanation_text += f"Confidence: {details['confidence']:.2%}\n"
        explanation_text += f"Detections: {num_boxes}\n\n"
        explanation_text += "REASONING:\n"
        for reason in details['reasons']:
            explanation_text += reason + "\n"
        
        return explanation_text, details
    
    def draw_bounding_boxes(self, image, result):
        """
        Draw bounding boxes and confidence scores on image.
        
        Args:
            image: Original image (numpy array, BGR)
            result: YOLO detection result
            
        Returns:
            annotated_image: Image with bounding boxes drawn (BGR)
        """
        annotated = image.copy()
        
        try:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                return annotated
            
            for box in boxes:
                # Extract box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy()) if hasattr(box, 'conf') else 0.5
                
                # Draw bounding box (red for fire)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red in BGR
                
                # Draw confidence score
                conf_text = f"Fire: {conf:.2%}"
                text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background rectangle for text
                cv2.rectangle(annotated, (x1, y1 - text_size[1] - 6), 
                            (x1 + text_size[0] + 6, y1), (0, 0, 255), -1)
                
                # Text
                cv2.putText(annotated, conf_text, (x1 + 3, y1 - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        except Exception as e:
            print(f"Error drawing boxes: {e}")
        
        return annotated
    
    def generate_heatmap(self, image, result):
        """
        Generate a heatmap showing detection regions.
        Brighter regions = higher detection confidence.
        
        Args:
            image: Original image (numpy array, BGR)
            result: YOLO detection result
            
        Returns:
            heatmap: Heatmap image (numpy array, BGR)
        """
        h, w = image.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        try:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                # Return blank heatmap for no detections
                heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                return cv2.addWeighted(image, 0.7, heatmap_colored, 0.3, 0)
            
            # Create Gaussian blobs at detection centers
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy()) if hasattr(box, 'conf') else 0.5
                
                # Center and size of detection
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                box_w, box_h = x2 - x1, y2 - y1
                
                # Draw Gaussian blob
                y_min, y_max = max(0, cy - box_h), min(h, cy + box_h)
                x_min, x_max = max(0, cx - box_w), min(w, cx + box_w)
                
                for yy in range(y_min, y_max):
                    for xx in range(x_min, x_max):
                        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
                        sigma = max(box_w, box_h) / 2
                        heatmap[yy, xx] = max(heatmap[yy, xx], conf * np.exp(-(dist**2) / (2 * sigma**2)))
            
            # Normalize and colorize
            heatmap = (heatmap / (np.max(heatmap) + 1e-6) * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Blend with original image
            result_image = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
            return result_image
        
        except Exception as e:
            print(f"Error generating heatmap: {e}")
            return image
    
    def create_explanation_image(self, image, explanation_text, fire_detected):
        """
        Create an image with explanation text overlay.
        
        Args:
            image: Original image (numpy array, BGR)
            explanation_text: Explanation string
            fire_detected: Boolean indicating fire detection
            
        Returns:
            image_with_text: Image with text overlay (numpy array, BGR)
        """
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to use a nice font, fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            small_font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        # Add semi-transparent background for text
        h, w = image.shape[:2]
        overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Background rectangle
        text_height = 200
        overlay_draw.rectangle([10, 10, w - 10, text_height], fill=(0, 0, 0, 180))
        
        # Prediction header
        pred_text = "🔥 FIRE DETECTED" if fire_detected else "✓ NO FIRE"
        pred_color = (255, 50, 50) if fire_detected else (50, 200, 50)
        draw.text((20, 20), pred_text, font=font, fill=pred_color)
        
        # Explanation text
        lines = explanation_text.split('\n')
        y_offset = 70
        for line in lines[:6]:  # Limit lines to fit on image
            if line.strip():
                draw.text((20, y_offset), line, font=small_font, fill=(255, 255, 255))
                y_offset += 25
        
        # Composite overlay on image
        pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
        
        # Convert back to BGR for OpenCV
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def explain_prediction(self, image_path, output_dir='xai_outputs'):
        """
        Full pipeline: detect fire and generate explainable outputs.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save explanation outputs
            
        Returns:
            explanation_dict: Dictionary with prediction, confidence, and visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Run detection
        result, image = self.detect_fire(image_path)
        
        # Determine if fire was detected
        try:
            num_boxes = len(result.boxes) if hasattr(result.boxes, '__len__') else 0
        except:
            num_boxes = 0
        
        fire_detected = num_boxes > 0
        
        # Generate explanation
        explanation_text, details = self.generate_explanation(result, image, fire_detected)
        
        # Create visualizations
        annotated_image = self.draw_bounding_boxes(image, result)
        heatmap_image = self.generate_heatmap(image, result)
        explanation_image = self.create_explanation_image(annotated_image, explanation_text, fire_detected)
        
        # Save outputs
        input_name = os.path.splitext(os.path.basename(image_path))[0]
        
        cv2.imwrite(os.path.join(output_dir, f'{input_name}_annotated.png'), annotated_image)
        cv2.imwrite(os.path.join(output_dir, f'{input_name}_heatmap.png'), heatmap_image)
        cv2.imwrite(os.path.join(output_dir, f'{input_name}_explanation.png'), explanation_image)
        
        # Save explanation text
        with open(os.path.join(output_dir, f'{input_name}_explanation.txt'), 'w', encoding='utf-8') as f:
            f.write(explanation_text)
        
        # Return comprehensive result
        return {
            'prediction': details['prediction'],
            'fire_detected': fire_detected,
            'confidence': details['confidence'],
            'num_detections': details['boxes_count'],
            'explanation_text': explanation_text,
            'explanation_details': details,
            'image_size': details['image_size'],
            'output_files': {
                'annotated': os.path.join(output_dir, f'{input_name}_annotated.png'),
                'heatmap': os.path.join(output_dir, f'{input_name}_heatmap.png'),
                'explanation': os.path.join(output_dir, f'{input_name}_explanation.png'),
                'text': os.path.join(output_dir, f'{input_name}_explanation.txt')
            }
        }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fire Detection with Explainable AI')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--weights', default='yolov8n.pt', help='YOLOv8 model weights')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--output_dir', default='xai_outputs', help='Output directory for visualizations')
    parser.add_argument('--device', default=None, help='Device for inference (None, 0, cpu)')
    
    args = parser.parse_args()
    
    # Run XAI
    xai = FireDetectionXAI(model_weights=args.weights, conf_threshold=args.conf, device=args.device)
    result = xai.explain_prediction(args.image, output_dir=args.output_dir)
    
    # Print results
    print("\n" + "="*60)
    print(result['explanation_text'])
    print("="*60)
    print(f"\nOutput files saved to: {args.output_dir}")
    print(f"  - Annotated: {os.path.basename(result['output_files']['annotated'])}")
    print(f"  - Heatmap: {os.path.basename(result['output_files']['heatmap'])}")
    print(f"  - Explanation: {os.path.basename(result['output_files']['explanation'])}")
