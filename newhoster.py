from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from newclass import RecyclableClassificationAPI
import os
import uuid
from datetime import datetime
from typing import Dict, Any

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize classifier
classifier_api = RecyclableClassificationAPI()

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Homepage with API documentation"""
    return """
    <h1>Recyclable Waste Classification API</h1>
    <h2>Endpoints:</h2>
    <ul>
        <li><b>POST /classify</b> - Classify waste in an image</li>
        <li><b>GET /categories</b> - List supported waste categories</li>
        <li><b>POST /batch_classify</b> - Classify multiple images</li>
    </ul>
    """

@app.route('/classify', methods=['POST'])
def classify_image() -> Dict[str, Any]:
    """
    Classify waste in a single image
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: image
        in: formData
        type: file
        required: true
        description: The image to classify
      - name: confidence
        in: formData
        type: number
        required: false
        description: Confidence threshold (0.1-0.9)
    responses:
      200:
        description: Classification results
      400:
        description: Invalid input
    """
    try:
        # Check if image file was uploaded
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'error': 'No image file provided',
                'timestamp': datetime.now().isoformat()
            }), 400

        file = request.files['image']
        
        # Check if file has a name
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'error': 'No file selected',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Save the uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Get confidence threshold (default 0.25)
        confidence = float(request.form.get('confidence', 0.25))

        # Classify the image
        result = classifier_api.classify_image(filepath, confidence_threshold=confidence)
        
        # Clean up - remove uploaded file
        try:
            os.remove(filepath)
        except Exception as e:
            app.logger.warning(f"Could not remove temp file {filepath}: {e}")

        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Error in classify_image: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/categories', methods=['GET'])
def get_categories() -> Dict[str, Any]:
    """
    Get list of supported waste categories
    ---
    responses:
      200:
        description: List of supported categories
    """
    return jsonify({
        'status': 'success',
        'categories': classifier_api.get_supported_objects(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/batch_classify', methods=['POST'])
def batch_classify() -> Dict[str, Any]:
    """
    Classify waste in multiple images
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: images
        in: formData
        type: file
        required: true
        description: The images to classify (multiple)
        multiple: true
      - name: confidence
        in: formData
        type: number
        required: false
        description: Confidence threshold (0.1-0.9)
    responses:
      200:
        description: Batch classification results
      400:
        description: Invalid input
    """
    try:
        # Check if files were uploaded
        if 'images' not in request.files:
            return jsonify({
                'status': 'error',
                'error': 'No image files provided',
                'timestamp': datetime.now().isoformat()
            }), 400

        files = request.files.getlist('images')
        
        # Check if any files were selected
        if not files or all(f.filename == '' for f in files):
            return jsonify({
                'status': 'error',
                'error': 'No files selected',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Get confidence threshold
        confidence = float(request.form.get('confidence', 0.25))
        
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    # Save file
                    filename = secure_filename(file.filename)
                    unique_filename = f"{uuid.uuid4()}_{filename}"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                    file.save(filepath)
                    
                    # Classify
                    result = classifier_api.classify_image(filepath, confidence_threshold=confidence)
                    result['filename'] = filename
                    results.append(result)
                    
                    # Clean up
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        app.logger.warning(f"Could not remove temp file {filepath}: {e}")
                        
                except Exception as e:
                    results.append({
                        'status': 'error',
                        'error': str(e),
                        'filename': file.filename,
                        'timestamp': datetime.now().isoformat()
                    })

        return jsonify({
            'status': 'success',
            'results': results,
            'total_images': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error in batch_classify: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/add_mapping', methods=['POST'])
def add_mapping() -> Dict[str, Any]:
    """
    Add custom object-to-recyclability mapping
    ---
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            object_class:
              type: string
              description: Name of the object to map
            recyclability:
              type: string
              description: Recyclability category
    responses:
      200:
        description: Mapping added successfully
      400:
        description: Invalid input
    """
    try:
        data = request.get_json()
        object_class = data.get('object_class')
        recyclability = data.get('recyclability')
        
        if not object_class or not recyclability:
            return jsonify({
                'status': 'error',
                'error': 'Both object_class and recyclability are required',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Define valid recyclability categories
        valid_categories = ['Recyclable', 'Non-Recyclable', 'Compostable', 'Special Handling', 'Unknown']
        
        if recyclability not in valid_categories:
            return jsonify({
                'status': 'error',
                'error': f'Invalid recyclability category. Must be one of: {", ".join(valid_categories)}',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        classifier_api.add_mapping(object_class, recyclability)
        
        return jsonify({
            'status': 'success',
            'message': f'Added mapping: {object_class} -> {recyclability}',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 400

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'status': 'error',
        'error': 'File too large. Maximum size is 16MB',
        'timestamp': datetime.now().isoformat()
    }), 413

if __name__ == '__main__':
    # Run the server
    print("Starting Waste Classification API Server...")
    print("Available endpoints:")
    print("  GET  /             - API documentation")
    print("  POST /classify      - Classify single image")
    print("  GET  /categories    - List supported categories")
    print("  POST /batch_classify - Classify multiple images")
    print("  POST /add_mapping   - Add custom recyclability mapping")
    print("\nServer running on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)