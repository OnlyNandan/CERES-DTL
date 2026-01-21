"""
Disease Detection Module - Plant Disease Recognition via Image Analysis
Uses TensorFlow/Keras for image classification
"""

import os
import io
import base64
import numpy as np
from PIL import Image
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Disease information database with treatments
DISEASE_DATABASE = {
    'apple_scab': {
        'name': 'Apple Scab',
        'name_hi': 'सेब का पपड़ी रोग',
        'name_kn': 'ಸೇಬು ಸ್ಕ್ಯಾಬ್',
        'severity': 'moderate',
        'symptoms': 'Dark olive-green spots on leaves, velvety texture, leaves may curl',
        'treatment': [
            'Apply fungicide (Mancozeb/Captan) at 2g/L',
            'Remove and destroy infected leaves',
            'Improve air circulation by pruning',
            'Apply sulfur-based sprays preventively'
        ],
        'prevention': 'Plant resistant varieties, maintain good sanitation'
    },
    'bacterial_spot': {
        'name': 'Bacterial Spot',
        'name_hi': 'जीवाणु धब्बा',
        'name_kn': 'ಬ್ಯಾಕ್ಟೀರಿಯಾ ಚುಕ್ಕೆ',
        'severity': 'high',
        'symptoms': 'Water-soaked spots, turning brown with yellow halos',
        'treatment': [
            'Apply copper-based bactericide',
            'Remove infected plant parts',
            'Avoid overhead irrigation',
            'Use disease-free seeds'
        ],
        'prevention': 'Crop rotation, resistant varieties'
    },
    'late_blight': {
        'name': 'Late Blight',
        'name_hi': 'पछेती अंगमारी',
        'name_kn': 'ತಡವಾದ ಬ್ಲೈಟ್',
        'severity': 'severe',
        'symptoms': 'Water-soaked lesions, white mold growth, rapid plant collapse',
        'treatment': [
            'Apply Metalaxyl + Mancozeb immediately',
            'Remove and burn infected plants',
            'Improve drainage',
            'Spray every 7-10 days during outbreak'
        ],
        'prevention': 'Plant certified disease-free tubers, proper spacing'
    },
    'early_blight': {
        'name': 'Early Blight',
        'name_hi': 'अगेती अंगमारी',
        'name_kn': 'ಆರಂಭಿಕ ಬ್ಲೈಟ್',
        'severity': 'moderate',
        'symptoms': 'Target-shaped concentric rings on leaves, yellowing',
        'treatment': [
            'Apply Chlorothalonil or Mancozeb',
            'Remove lower infected leaves',
            'Mulch to prevent soil splash',
            'Maintain adequate nitrogen levels'
        ],
        'prevention': 'Crop rotation, stake plants for airflow'
    },
    'leaf_mold': {
        'name': 'Leaf Mold',
        'name_hi': 'पत्ती फफूंद',
        'name_kn': 'ಎಲೆ ಶಿಲೀಂಧ್ರ',
        'severity': 'moderate',
        'symptoms': 'Yellow spots on upper leaf, olive-green mold below',
        'treatment': [
            'Improve ventilation',
            'Reduce humidity below 85%',
            'Apply fungicide if severe',
            'Remove infected leaves'
        ],
        'prevention': 'Proper spacing, avoid overhead watering'
    },
    'powdery_mildew': {
        'name': 'Powdery Mildew',
        'name_hi': 'चूर्णी फफूंदी',
        'name_kn': 'ಪುಡಿ ಶಿಲೀಂಧ್ರ',
        'severity': 'moderate',
        'symptoms': 'White powdery coating on leaves, stems, buds',
        'treatment': [
            'Apply sulfur-based fungicide',
            'Spray potassium bicarbonate solution',
            'Neem oil spray (2ml/L)',
            'Remove severely infected parts'
        ],
        'prevention': 'Adequate spacing, avoid excessive nitrogen'
    },
    'rust': {
        'name': 'Rust Disease',
        'name_hi': 'रस्ट रोग',
        'name_kn': 'ತುಕ್ಕು ರೋಗ',
        'severity': 'moderate',
        'symptoms': 'Orange-brown pustules on undersides of leaves',
        'treatment': [
            'Apply Propiconazole or Tebuconazole',
            'Remove alternate hosts',
            'Destroy crop residues',
            'Apply at first sign of disease'
        ],
        'prevention': 'Resistant varieties, crop rotation'
    },
    'septoria_leaf_spot': {
        'name': 'Septoria Leaf Spot',
        'name_hi': 'सेप्टोरिया पत्ती धब्बा',
        'name_kn': 'ಸೆಪ್ಟೋರಿಯಾ ಎಲೆ ಚುಕ್ಕೆ',
        'severity': 'moderate',
        'symptoms': 'Small circular spots with dark borders and tan centers',
        'treatment': [
            'Apply Chlorothalonil fungicide',
            'Remove infected lower leaves',
            'Avoid wetting foliage',
            'Mulch around plants'
        ],
        'prevention': 'Crop rotation, proper plant spacing'
    },
    'yellow_leaf_curl_virus': {
        'name': 'Yellow Leaf Curl Virus',
        'name_hi': 'पीला पत्ता मोड़ विषाणु',
        'name_kn': 'ಹಳದಿ ಎಲೆ ಸುರುಳಿ ವೈರಸ್',
        'severity': 'severe',
        'symptoms': 'Upward curling, yellowing leaves, stunted growth',
        'treatment': [
            'No cure - remove infected plants',
            'Control whitefly vectors with Imidacloprid',
            'Use yellow sticky traps',
            'Apply neem oil to control vectors'
        ],
        'prevention': 'Resistant varieties, reflective mulches, screen nurseries'
    },
    'healthy': {
        'name': 'Healthy Plant',
        'name_hi': 'स्वस्थ पौधा',
        'name_kn': 'ಆರೋಗ್ಯಕರ ಸಸ್ಯ',
        'severity': 'none',
        'symptoms': 'No visible disease symptoms',
        'treatment': ['Continue regular care and monitoring'],
        'prevention': 'Maintain current practices'
    },
    'rice_blast': {
        'name': 'Rice Blast',
        'name_hi': 'धान का ब्लास्ट',
        'name_kn': 'ಭತ್ತದ ಬ್ಲಾಸ್ಟ್',
        'severity': 'severe',
        'symptoms': 'Diamond-shaped lesions on leaves, neck rot',
        'treatment': [
            'Apply Tricyclazole (0.6g/L) or Isoprothiolane',
            'Drain excess water from fields',
            'Reduce nitrogen application',
            'Apply silicon-based fertilizers'
        ],
        'prevention': 'Resistant varieties, balanced fertilization'
    },
    'brown_spot': {
        'name': 'Brown Spot',
        'name_hi': 'भूरा धब्बा',
        'name_kn': 'ಕಂದು ಚುಕ್ಕೆ',
        'severity': 'moderate',
        'symptoms': 'Brown oval spots with gray center on leaves',
        'treatment': [
            'Apply Mancozeb or Carbendazim',
            'Improve soil fertility',
            'Proper water management',
            'Seed treatment with fungicide'
        ],
        'prevention': 'Balanced nutrition, proper drainage'
    },
    'bacterial_leaf_blight': {
        'name': 'Bacterial Leaf Blight',
        'name_hi': 'जीवाणु पत्ती झुलसा',
        'name_kn': 'ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಎಲೆ ಬ್ಲೈಟ್',
        'severity': 'severe',
        'symptoms': 'Water-soaked lesions turning yellow-white, wilting',
        'treatment': [
            'Apply Streptomycin sulfate',
            'Copper hydroxide spray',
            'Drain fields during infection',
            'Remove infected plant debris'
        ],
        'prevention': 'Resistant varieties, balanced fertilizer'
    },
    'cotton_leaf_curl': {
        'name': 'Cotton Leaf Curl Virus',
        'name_hi': 'कपास पत्ती मोड़ रोग',
        'name_kn': 'ಹತ್ತಿ ಎಲೆ ಸುರುಳಿ',
        'severity': 'severe',
        'symptoms': 'Upward/downward leaf curling, vein thickening, stunting',
        'treatment': [
            'Remove and destroy infected plants',
            'Control whitefly with Thiamethoxam',
            'Use neem-based insecticides',
            'Install yellow sticky traps'
        ],
        'prevention': 'Bt cotton varieties, early sowing, vector control'
    }
}

# Mapping of model class indices to disease keys
CLASS_MAPPING = {
    0: 'apple_scab',
    1: 'bacterial_spot',
    2: 'early_blight',
    3: 'late_blight',
    4: 'leaf_mold',
    5: 'powdery_mildew',
    6: 'rust',
    7: 'septoria_leaf_spot',
    8: 'yellow_leaf_curl_virus',
    9: 'healthy',
    10: 'rice_blast',
    11: 'brown_spot',
    12: 'bacterial_leaf_blight',
    13: 'cotton_leaf_curl'
}


class DiseaseDetector:
    """Plant disease detection using image analysis"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.image_size = (224, 224)
        
    def load_model(self) -> bool:
        """Load the disease detection model"""
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'disease_model.h5')
        
        # Check if TensorFlow model exists
        if os.path.exists(model_path):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path)
                self.model_loaded = True
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        
        # If no TF model, we'll use rule-based detection
        self.model_loaded = False
        return False
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess image for model input"""
        try:
            # Open image from bytes
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize
            image = image.resize(self.image_size)
            
            # Convert to array and normalize
            img_array = np.array(image) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {e}")
    
    def analyze_image_colors(self, image_data: bytes) -> Dict[str, Any]:
        """Analyze image colors for rule-based detection"""
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for analysis
            image = image.resize((100, 100))
            img_array = np.array(image)
            
            # Calculate color statistics
            r_mean = np.mean(img_array[:, :, 0])
            g_mean = np.mean(img_array[:, :, 1])
            b_mean = np.mean(img_array[:, :, 2])
            
            # Calculate yellow/brown indicators
            yellow_ratio = (r_mean + g_mean) / (2 * max(b_mean, 1))
            green_ratio = g_mean / max(r_mean, 1)
            brown_ratio = r_mean / max(g_mean + b_mean, 1)
            
            # Standard deviations for texture analysis
            r_std = np.std(img_array[:, :, 0])
            g_std = np.std(img_array[:, :, 1])
            
            return {
                'r_mean': r_mean,
                'g_mean': g_mean,
                'b_mean': b_mean,
                'yellow_ratio': yellow_ratio,
                'green_ratio': green_ratio,
                'brown_ratio': brown_ratio,
                'r_std': r_std,
                'g_std': g_std,
                'texture_variance': (r_std + g_std) / 2
            }
        except Exception as e:
            return {'error': str(e)}
    
    def detect_disease_rule_based(self, image_data: bytes) -> Dict[str, Any]:
        """Rule-based disease detection when model is not available"""
        colors = self.analyze_image_colors(image_data)
        
        if 'error' in colors:
            return {
                'success': False,
                'error': colors['error']
            }
        
        # Initialize scores for different conditions
        scores = {}
        
        # Extract color metrics
        yellow_ratio = colors['yellow_ratio']
        green_ratio = colors['green_ratio']
        brown_ratio = colors['brown_ratio']
        texture_variance = colors['texture_variance']
        r_mean = colors['r_mean']
        g_mean = colors['g_mean']
        b_mean = colors['b_mean']
        
        # Calculate additional metrics
        brightness = (r_mean + g_mean + b_mean) / 3
        saturation = max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)
        
        # HEALTHY: Strong green, low texture variance, good brightness
        if green_ratio > 1.1 and texture_variance < 30 and saturation > 40:
            scores['healthy'] = 0.7 + (green_ratio - 1.1) * 0.2
        elif green_ratio > 1.0 and texture_variance < 35:
            scores['healthy'] = 0.5
        
        # YELLOW LEAF CURL VIRUS: Very high yellow, curling indicators (unusual aspect)
        if yellow_ratio > 2.0 and green_ratio < 0.9:
            scores['yellow_leaf_curl_virus'] = 0.5 + (yellow_ratio - 2.0) * 0.1
        elif yellow_ratio > 1.8 and texture_variance > 35:
            scores['yellow_leaf_curl_virus'] = 0.45
        
        # EARLY BLIGHT: Target-shaped spots, moderate yellow/brown
        if yellow_ratio > 1.3 and yellow_ratio < 1.8 and texture_variance > 30:
            scores['early_blight'] = 0.55
        
        # LATE BLIGHT: Water-soaked lesions, darker areas
        if brightness < 100 and texture_variance > 40 and brown_ratio > 0.6:
            scores['late_blight'] = 0.6
        elif texture_variance > 45 and green_ratio < 1.0:
            scores['late_blight'] = 0.5
        
        # BROWN SPOT: Obvious brown areas with texture
        if brown_ratio > 0.9 and texture_variance > 25:
            scores['brown_spot'] = 0.65
        elif brown_ratio > 0.7 and texture_variance > 35:
            scores['brown_spot'] = 0.5
        
        # SEPTORIA LEAF SPOT: Small spots, moderate texture
        if texture_variance > 30 and texture_variance < 45 and brown_ratio > 0.5 and brown_ratio < 0.8:
            scores['septoria_leaf_spot'] = 0.55
        
        # RUST: Orange-brown pustules
        if r_mean > g_mean and r_mean > 120 and brown_ratio > 0.6 and brown_ratio < 0.9:
            scores['rust'] = 0.55
        
        # POWDERY MILDEW: Whitish appearance, high brightness
        if brightness > 160 and saturation < 50 and green_ratio < 1.1:
            scores['powdery_mildew'] = 0.6
        elif brightness > 140 and saturation < 40:
            scores['powdery_mildew'] = 0.45
        
        # LEAF MOLD: Yellow spots with darker underside patterns
        if yellow_ratio > 1.2 and yellow_ratio < 1.6 and texture_variance > 25:
            scores['leaf_mold'] = 0.5
        
        # BACTERIAL SPOT: Water-soaked spots with yellow halos
        if texture_variance > 35 and yellow_ratio > 1.1 and green_ratio < 1.0:
            scores['bacterial_spot'] = 0.5
        
        # BACTERIAL LEAF BLIGHT: Similar to bacterial spot but more severe
        if texture_variance > 40 and yellow_ratio > 1.3 and brightness < 130:
            scores['bacterial_leaf_blight'] = 0.55
        
        # RICE BLAST: Diamond-shaped lesions
        if texture_variance > 35 and brown_ratio > 0.5 and green_ratio > 0.7:
            scores['rice_blast'] = 0.45
        
        # If no significant detection, lean towards healthy with moderate confidence
        if not scores or max(scores.values()) < 0.4:
            scores['healthy'] = max(scores.get('healthy', 0), 0.55)
        
        # Get top prediction
        if scores:
            top_disease = max(scores, key=scores.get)
            confidence = min(scores[top_disease] * 100, 85)  # Cap at 85% for rule-based
        else:
            top_disease = 'healthy'
            confidence = 50.0
        
        disease_info = DISEASE_DATABASE.get(top_disease, DISEASE_DATABASE['healthy'])
        
        return {
            'success': True,
            'disease_key': top_disease,
            'disease_name': disease_info['name'],
            'confidence': confidence,
            'severity': disease_info['severity'],
            'symptoms': disease_info['symptoms'],
            'treatment': disease_info['treatment'],
            'prevention': disease_info['prevention'],
            'all_scores': scores,
            'method': 'rule_based',
            'color_analysis': colors
        }
    
    def detect_disease(self, image_data: bytes, language: str = 'en') -> Dict[str, Any]:
        """
        Main disease detection method
        
        Args:
            image_data: Raw image bytes
            language: Language code for response
            
        Returns:
            Detection results with disease info and treatment
        """
        try:
            # Try ML model first
            if self.model_loaded and self.model is not None:
                img_array = self.preprocess_image(image_data)
                predictions = self.model.predict(img_array, verbose=0)
                
                # Get top prediction
                top_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][top_idx]) * 100
                
                disease_key = CLASS_MAPPING.get(top_idx, 'healthy')
                disease_info = DISEASE_DATABASE.get(disease_key, DISEASE_DATABASE['healthy'])
                
                # Get localized name
                name_key = f'name_{language}' if language != 'en' else 'name'
                disease_name = disease_info.get(name_key, disease_info['name'])
                
                # Get top 3 predictions
                top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                alternatives = []
                for idx in top_3_idx[1:]:
                    alt_key = CLASS_MAPPING.get(idx, 'healthy')
                    alt_info = DISEASE_DATABASE.get(alt_key, DISEASE_DATABASE['healthy'])
                    alternatives.append({
                        'disease': alt_info['name'],
                        'confidence': float(predictions[0][idx]) * 100
                    })
                
                return {
                    'success': True,
                    'disease_key': disease_key,
                    'disease_name': disease_name,
                    'confidence': confidence,
                    'severity': disease_info['severity'],
                    'symptoms': disease_info['symptoms'],
                    'treatment': disease_info['treatment'],
                    'prevention': disease_info['prevention'],
                    'alternatives': alternatives,
                    'method': 'ml_model'
                }
            else:
                # Fallback to rule-based detection
                return self.detect_disease_rule_based(image_data)
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Error analyzing image. Please try again with a clearer image.'
            }
    
    def get_disease_info(self, disease_key: str, language: str = 'en') -> Dict[str, Any]:
        """Get detailed information about a specific disease"""
        if disease_key not in DISEASE_DATABASE:
            return {'success': False, 'error': 'Disease not found'}
        
        info = DISEASE_DATABASE[disease_key]
        name_key = f'name_{language}' if language != 'en' else 'name'
        
        return {
            'success': True,
            'disease_key': disease_key,
            'name': info.get(name_key, info['name']),
            'severity': info['severity'],
            'symptoms': info['symptoms'],
            'treatment': info['treatment'],
            'prevention': info['prevention']
        }
    
    def get_all_diseases(self, language: str = 'en') -> list:
        """Get list of all detectable diseases"""
        diseases = []
        name_key = f'name_{language}' if language != 'en' else 'name'
        
        for key, info in DISEASE_DATABASE.items():
            if key != 'healthy':
                diseases.append({
                    'key': key,
                    'name': info.get(name_key, info['name']),
                    'severity': info['severity']
                })
        
        return sorted(diseases, key=lambda x: x['name'])


# Singleton instance
disease_detector = DiseaseDetector()
