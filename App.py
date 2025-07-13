from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, resnet34
import cv2
import numpy as np
from PIL import Image
import base64
import io
import os
from pathlib import Path
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Model configuration
MODEL_PATH = 'anti_overfitting_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names from your model
CLASS_NAMES = ['1_Microgram', '1_Nanogram', '10_Nanogram', '100_Nanogram']

class ImprovedLFAPreprocessor:
    """Enhanced preprocessing with normalization - same as your training code"""

    def enhance_image(self, image):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert RGB to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_clahe = clahe.apply(l)
        
        # Merge channels and convert back to RGB
        enhanced_lab = cv2.merge([l_clahe, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        return enhanced_rgb

class AntiOverfittingLFAClassifier(nn.Module):
    """Classifier designed to prevent overfitting - same as your training code"""
    
    def __init__(self, num_classes=4, model_name='efficientnet_b0', dropout_rate=0.7):
        super(AntiOverfittingLFAClassifier, self).__init__()
        
        # Use smaller, less complex models
        if model_name == 'efficientnet_b0':
            self.backbone = efficientnet_b0(pretrained=True)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:  # resnet34
            self.backbone = resnet34(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        # Freeze more layers to prevent overfitting
        self._freeze_layers()
        
        # Simpler classifier with heavy regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(64, num_classes)
        )
        
        # Apply weight initialization
        self._init_weights()
    
    def _freeze_layers(self):
        """Freeze most of the backbone layers"""
        # Freeze first 80% of parameters
        total_params = list(self.backbone.parameters())
        freeze_count = int(0.8 * len(total_params))
        
        for param in total_params[:freeze_count]:
            param.requires_grad = False
    
    def _init_weights(self):
        """Initialize weights properly"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

def get_test_transform(image_size=224):
    """Get the same test transform used during training"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class LFAPredictor:
    """Main prediction class that handles model loading and inference"""
    
    def __init__(self, model_path, device):
        self.device = device
        self.model = None
        self.preprocessor = ImprovedLFAPreprocessor()
        self.transform = get_test_transform()
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Create model architecture
            self.model = AntiOverfittingLFAClassifier(
                num_classes=4, 
                model_name='efficientnet_b0', 
                dropout_rate=0.7
            )
            
            # Load trained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully from {model_path}")
            print(f"Model validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image):
        """Apply the same preprocessing as during training"""
        try:
            # Convert to PIL if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply CLAHE enhancement
            enhanced_image = self.preprocessor.enhance_image(image)
            enhanced_pil = Image.fromarray(enhanced_image)
            
            # Apply transforms
            transformed_image = self.transform(enhanced_pil)
            
            return transformed_image.unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            raise
    
    def predict(self, image):
        """Make prediction on a single image"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            processed_image = processed_image.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(processed_image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Get all class probabilities
                class_probs = probabilities[0].cpu().numpy()
                
                result = {
                    'predicted_class': CLASS_NAMES[predicted.item()],
                    'predicted_class_index': predicted.item(),
                    'confidence': confidence.item(),
                    'class_probabilities': {
                        CLASS_NAMES[i]: float(prob) for i, prob in enumerate(class_probs)
                    }
                }
                
                return result
                
        except Exception as e:
            print(f"Error in prediction: {e}")
            raise

# Initialize predictor
predictor = None

def initialize_model():
    """Initialize the model predictor"""
    global predictor
    try:
        predictor = LFAPredictor(MODEL_PATH, DEVICE)
        print("✅ Model initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        predictor = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'device': str(DEVICE),
        'model_path': MODEL_PATH
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if predictor is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        # Check if image is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        image_file = request.files['file']
        
        # Check if file is selected
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and process image
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        import time

        start_time = time.time()

        # Make prediction
        result = predictor.predict(image)

        end_time = time.time()
        processing_time = round(end_time - start_time, 3)

        # Convert class_probabilities to array
        confidence_scores = list(result['class_probabilities'].values())

        # Add frontend-compatible fields
        result['confidence_scores'] = confidence_scores
        result['model_used'] = "EfficientNet-B0"
        result['processing_time'] = processing_time
        result['image_quality'] = 0.92  # Placeholder – optionally calculate dynamically
        result['image_type'] = image.mode  # RGB, etc.

        # Optional: preserve original metadata
        result['metadata'] = {
            'image_size': image.size,
            'image_mode': image.mode,
            'filename': image_file.filename
        }

        return jsonify(result)

        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Prediction endpoint for base64 encoded images"""
    try:
        if predictor is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        # Get JSON data
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No base64 image provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Make prediction
        result = predictor.predict(image)
        
        # Add metadata
        result['metadata'] = {
            'image_size': image.size,
            'image_mode': image.mode,
            'encoding': 'base64'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if predictor is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        info = {
            'model_architecture': 'AntiOverfittingLFAClassifier',
            'backbone': 'EfficientNet-B0',
            'num_classes': len(CLASS_NAMES),
            'class_names': CLASS_NAMES,
            'input_size': [224, 224],
            'preprocessing': {
                'clahe_enhancement': True,
                'normalization': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            },
            'device': str(DEVICE)
        }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        if predictor is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        # Check if images are in request
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        image_files = request.files.getlist('images')
        
        if not image_files:
            return jsonify({'error': 'No images selected'}), 400
        
        results = []
        
        for i, image_file in enumerate(image_files):
            try:
                # Read and process image
                image_bytes = image_file.read()
                image = Image.open(io.BytesIO(image_bytes))
                
                # Make prediction
                result = predictor.predict(image)
                result['image_index'] = i
                result['filename'] = image_file.filename
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    'image_index': i,
                    'filename': image_file.filename,
                    'error': str(e)
                })
        
        return jsonify({'predictions': results})
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'Flask LFA Model API is running',
        'version': '1.0',
        'endpoints': {
            'health': '/health',
            'predict': '/predict',
            'predict_base64': '/predict_base64',
            'model_info': '/model_info',
            'predict_batch': '/predict_batch'
        }
    })


if __name__ == '__main__':
    print("🚀 Starting Flask LFA Model API...")
    print(f"📱 Device: {DEVICE}")
    print(f"📁 Model Path: {MODEL_PATH}")
    
    # Initialize model on startup
    initialize_model()
    
    if predictor is None:
        print("⚠️  WARNING: Model not loaded! Check if model file exists.")
    
    # Run Flask app
    print("🌐 Starting server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)