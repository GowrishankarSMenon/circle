from newclass import RecyclableClassificationAPI
import os

def test_classifier(image_folder="waste images"):
    """Test the classifier with all images in a folder"""
    api = RecyclableClassificationAPI()
    
    print("=== Starting Waste Classification Tests ===")
    
    # Test each image in the folder
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.png')):
            image_path = os.path.join(image_folder, filename)
            print(f"\nTesting image: {filename}")
            
            result = api.classify_image(image_path)
            
            if result['status'] == 'success':
                print(f"\nResults for {filename}:")
                print(f"Total objects detected: {result['total_objects_detected']}")
                
                # Print summary
                for category, data in result['summary'].items():
                    if data['count'] > 0:
                        items = [obj['detected_class'] for obj in data['items']]
                        print(f"  {category.title()}: {data['count']} ({', '.join(items)})")
                
                # Print assessment
                print(f"\nAssessment: {result['overall_assessment']['message']}")
                print(f"Recommendation: {result['overall_assessment']['recommendation']}")
            else:
                print(f"Error processing {filename}: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_classifier()