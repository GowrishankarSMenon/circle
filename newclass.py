# Recyclable vs Non-Recyclable Waste Classifier
# This system detects objects and classifies them as recyclable or non-recyclable

import cv2
import numpy as np
from PIL import Image
import json
import os
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecyclableClassifier:
    def __init__(self):
        """Initialize the recyclable classifier"""
        
        # Mapping of YOLO detected objects to recyclability
        self.recyclable_mapping = {
            # Definitely Recyclable
            'bottle': 'Recyclable',
            'wine glass': 'Recyclable',
            'cup': 'Recyclable',
            'fork': 'Recyclable',
            'knife': 'Recyclable',
            'spoon': 'Recyclable',
            'bowl': 'Recyclable',
            'tv': 'Recyclable',
            'laptop': 'Recyclable',
            'mouse': 'Recyclable',
            'keyboard': 'Recyclable',
            'cell phone': 'Recyclable',
            'microwave': 'Recyclable',
            'oven': 'Recyclable',
            'toaster': 'Recyclable',
            'refrigerator': 'Recyclable',
            'book': 'Recyclable',
            'vase': 'Recyclable',
            'scissors': 'Recyclable',
            'clock': 'Recyclable',
            'can': 'Recyclable',
            'jar': 'Recyclable',
            'container': 'Recyclable',
            'cardboard': 'Recyclable',
            'paper': 'Recyclable',
            'newspaper': 'Recyclable',
            'magazine': 'Recyclable',
            'aluminum': 'Recyclable',
            'metal': 'Recyclable',
            'glass': 'Recyclable',
            'plastic': 'Recyclable',
            'electronics': 'Recyclable',
            'wire': 'Recyclable',
            'cable': 'Recyclable',
            'battery': 'Recyclable',
            'computer': 'Recyclable',
            'monitor': 'Recyclable',
            'phone': 'Recyclable',
            'tablet': 'Recyclable',
            'camera': 'Recyclable',
            'radio': 'Recyclable',
            'speaker': 'Recyclable',
            'headphones': 'Recyclable',
            'charger': 'Recyclable',
            'remote': 'Recyclable',
            
            # Organic/Compostable (depends on local programs, generally not recyclable in standard recycling)
            'banana': 'Compostable',
            'apple': 'Compostable',
            'sandwich': 'Compostable',
            'orange': 'Compostable',
            'broccoli': 'Compostable',
            'carrot': 'Compostable',
            'hot dog': 'Compostable',
            'pizza': 'Compostable',
            'donut': 'Compostable',
            'cake': 'Compostable',
            'potted plant': 'Compostable',
            'food': 'Compostable',
            'fruit': 'Compostable',
            'vegetable': 'Compostable',
            'bread': 'Compostable',
            'meat': 'Compostable',
            'fish': 'Compostable',
            'egg': 'Compostable',
            'cheese': 'Compostable',
            'lettuce': 'Compostable',
            'tomato': 'Compostable',
            'potato': 'Compostable',
            'onion': 'Compostable',
            'garlic': 'Compostable',
            'leaf': 'Compostable',
            'flower': 'Compostable',
            'plant': 'Compostable',
            
            # Generally Non-Recyclable
            'teddy bear': 'Non-Recyclable',
            'backpack': 'Non-Recyclable',
            'umbrella': 'Non-Recyclable',
            'handbag': 'Non-Recyclable',
            'tie': 'Non-Recyclable',
            'suitcase': 'Non-Recyclable',
            'sports ball': 'Non-Recyclable',
            'frisbee': 'Non-Recyclable',
            'kite': 'Non-Recyclable',
            'chair': 'Non-Recyclable',
            'couch': 'Non-Recyclable',
            'bed': 'Non-Recyclable',
            'dining table': 'Non-Recyclable',
            'toilet': 'Non-Recyclable',
            'sink': 'Non-Recyclable',
            'toothbrush': 'Non-Recyclable',
            'hair drier': 'Non-Recyclable',
            'clothing': 'Non-Recyclable',
            'shoes': 'Non-Recyclable',
            'plastic bag': 'Non-Recyclable',  # Most places don't recycle plastic bags
            'styrofoam': 'Non-Recyclable',
            'ceramic': 'Non-Recyclable',
            'mirror': 'Non-Recyclable',
            'lightbulb': 'Non-Recyclable',
            'napkin': 'Non-Recyclable',
            'tissue': 'Non-Recyclable',
            'diaper': 'Non-Recyclable',
            'cigarette': 'Non-Recyclable',
            'gum': 'Non-Recyclable',
            
            # Special/Hazardous - Need special handling
            'battery': 'Special Handling',
            'paint can': 'Special Handling',
            'aerosol': 'Special Handling',
            'chemical': 'Special Handling',
            'motor oil': 'Special Handling',
            'tire': 'Special Handling',
            'mattress': 'Special Handling'
        }
    
    def detect_objects(self, image_path: str, confidence_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Detect objects in image using YOLO
        
        Args:
            image_path: Path to the image
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detected objects with classification
        """
        try:
            from ultralytics import YOLO
            
            # Load YOLO model
            model = YOLO('yolov8n.pt')
            
            # Run detection
            results = model(image_path, conf=confidence_threshold, verbose=False)
            
            detected_objects = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = model.names[class_id]
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Classify recyclability
                        recyclability = self._classify_recyclability(class_name)
                        
                        detected_objects.append({
                            'detected_class': class_name,
                            'confidence': round(confidence, 3),
                            'recyclability': recyclability,
                            'bbox': {
                                'x1': round(x1, 1),
                                'y1': round(y1, 1),
                                'x2': round(x2, 1),
                                'y2': round(y2, 1)
                            }
                        })
            
            # Sort by confidence
            detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
            
            return detected_objects
            
        except ImportError:
            logger.error("ultralytics not installed. Install with: pip install ultralytics")
            return []
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return []
    
    def _classify_recyclability(self, class_name: str) -> str:
        """
        Classify if an object is recyclable based on its class name
        
        Args:
            class_name: Name of the detected object class
            
        Returns:
            Recyclability classification
        """
        class_name_lower = class_name.lower()
        
        # Direct mapping
        if class_name_lower in self.recyclable_mapping:
            return self.recyclable_mapping[class_name_lower]
        
        # Keyword-based classification
        recyclable_keywords = [
            'bottle', 'can', 'jar', 'glass', 'metal', 'aluminum', 'steel',
            'paper', 'cardboard', 'book', 'magazine', 'newspaper',
            'plastic', 'container', 'electronic', 'computer', 'phone',
            'laptop', 'tv', 'monitor', 'keyboard', 'mouse', 'cable',
            'wire', 'battery', 'camera', 'radio', 'speaker'
        ]
        
        compostable_keywords = [
            'food', 'fruit', 'vegetable', 'plant', 'flower', 'leaf',
            'organic', 'apple', 'banana', 'orange', 'bread', 'meat',
            'fish', 'egg', 'cheese', 'pizza', 'sandwich'
        ]
        
        non_recyclable_keywords = [
            'clothing', 'fabric', 'leather', 'rubber', 'ceramic',
            'styrofoam', 'foam', 'tissue', 'napkin', 'diaper',
            'cigarette', 'gum', 'mirror', 'lightbulb'
        ]
        
        # Check keywords
        for keyword in recyclable_keywords:
            if keyword in class_name_lower:
                return 'Recyclable'
        
        for keyword in compostable_keywords:
            if keyword in class_name_lower:
                return 'Compostable'
        
        for keyword in non_recyclable_keywords:
            if keyword in class_name_lower:
                return 'Non-Recyclable'
        
        # Default to unknown if no match
        return 'Unknown'
    
    def classify_image(self, image_path: str, confidence_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Classify all objects in an image for recyclability
        
        Args:
            image_path: Path to the image
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Dictionary with detection and classification results
        """
        try:
            detected_objects = self.detect_objects(image_path, confidence_threshold)
            
            # Summarize results
            summary = {
                'recyclable': [],
                'non_recyclable': [],
                'compostable': [],
                'special_handling': [],
                'unknown': []
            }
            
            for obj in detected_objects:
                category = obj['recyclability'].lower().replace(' ', '_').replace('-', '_')
                if category not in summary:
                    summary[category] = []
                summary[category].append(obj)
            
            # Calculate statistics
            total_objects = len(detected_objects)
            recyclable_count = len(summary['recyclable'])
            non_recyclable_count = len(summary['non_recyclable'])
            compostable_count = len(summary['compostable'])
            special_handling_count = len(summary['special_handling'])
            unknown_count = len(summary['unknown'])
            
            return {
                'status': 'success',
                'image_path': image_path,
                'timestamp': datetime.now().isoformat(),
                'total_objects_detected': total_objects,
                'confidence_threshold': confidence_threshold,
                'detected_objects': detected_objects,
                'summary': {
                    'recyclable': {
                        'count': recyclable_count,
                        'items': summary['recyclable']
                    },
                    'non_recyclable': {
                        'count': non_recyclable_count,
                        'items': summary['non_recyclable']
                    },
                    'compostable': {
                        'count': compostable_count,
                        'items': summary['compostable']
                    },
                    'special_handling': {
                        'count': special_handling_count,
                        'items': summary['special_handling']
                    },
                    'unknown': {
                        'count': unknown_count,
                        'items': summary['unknown']
                    }
                },
                'overall_assessment': self._get_overall_assessment(
                    recyclable_count, non_recyclable_count, compostable_count, 
                    special_handling_count, unknown_count
                )
            }
            
        except Exception as e:
            logger.error(f"Error classifying image: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'detected_objects': [],
                'total_objects_detected': 0
            }
    
    def _get_overall_assessment(self, recyclable: int, non_recyclable: int, 
                               compostable: int, special: int, unknown: int) -> Dict[str, Any]:
        """Generate overall assessment of the waste in the image"""
        
        total = recyclable + non_recyclable + compostable + special + unknown
        
        if total == 0:
            return {
                'message': 'No waste items detected in the image',
                'recommendation': 'Try a clearer image with visible waste items'
            }
        
        recyclable_percentage = (recyclable / total) * 100
        compostable_percentage = (compostable / total) * 100
        
        if recyclable_percentage > 70:
            message = "Great! Most items in this image are recyclable."
            recommendation = "Make sure to clean containers and separate materials properly before recycling."
        elif compostable_percentage > 70:
            message = "Most items are organic waste suitable for composting."
            recommendation = "Consider composting these items if you have access to composting facilities."
        elif (recyclable_percentage + compostable_percentage) > 70:
            message = "Good news! Most items can be diverted from landfill."
            recommendation = "Separate recyclable and compostable items appropriately."
        else:
            message = "Several items may need special handling or disposal."
            recommendation = "Check local guidelines for proper disposal methods."
        
        return {
            'message': message,
            'recommendation': recommendation,
            'recyclable_percentage': round(recyclable_percentage, 1),
            'compostable_percentage': round(compostable_percentage, 1),
            'total_items': total
        }
    
    def add_custom_mapping(self, object_class: str, recyclability: str):
        """Add custom recyclability mapping"""
        valid_categories = ['Recyclable', 'Non-Recyclable', 'Compostable', 'Special Handling', 'Unknown']
        
        if recyclability in valid_categories:
            self.recyclable_mapping[object_class.lower()] = recyclability
            logger.info(f"Added mapping: {object_class} -> {recyclability}")
        else:
            logger.error(f"Invalid recyclability category. Must be one of: {valid_categories}")

# API wrapper for easy use
class RecyclableClassificationAPI:
    def __init__(self):
        """Initialize the API"""
        self.classifier = RecyclableClassifier()
    
    def classify_image(self, image_path: str, confidence_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Main API method to classify image for recyclability
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence for object detection
            
        Returns:
            Complete classification results
        """
        return self.classifier.classify_image(image_path, confidence_threshold)
    
    def get_supported_objects(self) -> List[str]:
        """Get list of objects that can be classified"""
        return list(self.classifier.recyclable_mapping.keys())
    
    def add_mapping(self, object_class: str, recyclability: str):
        """Add custom object-to-recyclability mapping"""
        return self.classifier.add_custom_mapping(object_class, recyclability)

# Example usage and testing
def test_recyclable_classifier():
    """Test the recyclable classifier with example images"""
    
    # Initialize API
    api = RecyclableClassificationAPI()
    
    # Test image path
    image_path = "./waste images/bottle.jpg"
    
    print("=== Recyclable Waste Classification Test ===")
    
    if os.path.exists(image_path):
        # Test with different confidence levels
        confidence_levels = [0.1, 0.25, 0.5]
        
        for conf in confidence_levels:
            print(f"\n--- Testing with confidence threshold: {conf} ---")
            
            result = api.classify_image(image_path, confidence_threshold=conf)
            
            if result['status'] == 'success':
                print(f"Total objects detected: {result['total_objects_detected']}")
                
                # Print detected objects
                if result['detected_objects']:
                    print("\nDetected objects:")
                    for obj in result['detected_objects']:
                        print(f"  - {obj['detected_class']} ({obj['confidence']}) -> {obj['recyclability']}")
                
                # Print summary
                print(f"\nSummary:")
                summary = result['summary']
                for category, data in summary.items():
                    if data['count'] > 0:
                        print(f"  {category.replace('_', ' ').title()}: {data['count']} items")
                
                # Print overall assessment
                assessment = result['overall_assessment']
                print(f"\nOverall Assessment:")
                print(f"  {assessment['message']}")
                print(f"  Recommendation: {assessment['recommendation']}")
                
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
    
    else:
        print(f"Test image not found: {image_path}")
        print("Please provide a valid image path to test the classifier.")

if __name__ == "__main__":
    test_recyclable_classifier()