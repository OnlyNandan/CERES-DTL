import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'ceres-dtl-secret-key-change-in-production')
    
    # WeatherAPI.com - Primary Weather Source (FREE tier: 1M calls/month)
    WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY', '6d50534dcffe47e4899171904260301')
    WEATHER_API_BASE = "https://api.weatherapi.com/v1"
    
    # FREE APIs - No keys required (Backup)
    OPEN_METEO_BASE = "https://api.open-meteo.com/v1"
    NOMINATIM_BASE = "https://nominatim.openstreetmap.org"
    
    # Optional APIs (free tiers available)
    DATA_GOV_API_KEY = os.environ.get('DATA_GOV_API_KEY', '')
    
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///ceres.db')
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # ML Model Paths
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'crop_model.pkl')
    YIELD_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'yield_model.pkl')
    PEST_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'pest_model.pkl')
    SOIL_HEALTH_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'soil_health_model.pkl')
    
    SUPPORTED_LANGUAGES = ['en', 'hi', 'kn', 'te', 'ta', 'mr']
    DEFAULT_LANGUAGE = 'en'
    
    # ==========================================
    # ADVANCED AGRICULTURAL ML CONSTANTS
    # ==========================================
    
    # Penman-Monteith Evapotranspiration Constants
    STEFAN_BOLTZMANN = 4.903e-9  # MJ K-4 m-2 day-1
    PSYCHROMETRIC_CONST = 0.067  # kPa/°C at sea level
    SOLAR_CONSTANT = 0.0820  # MJ m-2 min-1
    
    # Growing Degree Days (GDD) Base Temperatures (°C)
    GDD_BASE_TEMPS = {
        'rice': 10, 'wheat': 4, 'maize': 10, 'cotton': 15.5,
        'sugarcane': 12, 'potato': 7, 'tomato': 10, 'onion': 7,
        'groundnut': 13, 'soybean': 10, 'mustard': 5, 'chickpea': 5,
        'lentil': 5, 'jute': 15, 'tea': 13, 'coffee': 15,
        'pepper': 18, 'cardamom': 15, 'turmeric': 15, 'banana': 14,
        'mango': 15, 'coconut': 20, 'millets': 10
    }
    
    # Crop Coefficient (Kc) for ET calculation by growth stage (FAO-56)
    CROP_KC = {
        'rice': {'initial': 1.05, 'dev': 1.10, 'mid': 1.20, 'late': 0.90},
        'wheat': {'initial': 0.30, 'dev': 0.75, 'mid': 1.15, 'late': 0.40},
        'maize': {'initial': 0.30, 'dev': 0.80, 'mid': 1.20, 'late': 0.60},
        'cotton': {'initial': 0.35, 'dev': 0.75, 'mid': 1.20, 'late': 0.70},
        'sugarcane': {'initial': 0.40, 'dev': 0.80, 'mid': 1.25, 'late': 0.75},
        'potato': {'initial': 0.50, 'dev': 0.80, 'mid': 1.15, 'late': 0.75},
        'tomato': {'initial': 0.60, 'dev': 0.85, 'mid': 1.15, 'late': 0.80},
        'onion': {'initial': 0.70, 'dev': 0.85, 'mid': 1.05, 'late': 0.75},
        'groundnut': {'initial': 0.40, 'dev': 0.75, 'mid': 1.15, 'late': 0.60},
        'soybean': {'initial': 0.40, 'dev': 0.80, 'mid': 1.15, 'late': 0.50}
    }
    
    # Yield Impact Factors (multipliers)
    YIELD_FACTORS = {
        'temperature_stress': {'optimal': (20, 30), 'penalty_per_degree': 0.03},
        'water_stress': {'threshold': 0.5, 'penalty_rate': 0.4},
        'nutrient_deficiency': {'N': 0.3, 'P': 0.2, 'K': 0.15},
        'soil_ph': {'optimal': (6.0, 7.0), 'penalty_per_unit': 0.1}
    }
    
    # Disease Risk Models (temp in °C, humidity in %)
    DISEASE_MODELS = {
        'late_blight': {
            'crops': ['potato', 'tomato'],
            'temp_range': (10, 25), 'humidity_min': 80,
            'rain_trigger': 10, 'risk_formula': 'sporangia'
        },
        'powdery_mildew': {
            'crops': ['wheat', 'mango', 'grapes'],
            'temp_range': (20, 30), 'humidity_range': (40, 70),
            'risk_formula': 'linear'
        },
        'rust': {
            'crops': ['wheat', 'coffee'],
            'temp_range': (15, 25), 'humidity_min': 70,
            'dew_hours': 6, 'risk_formula': 'infection_period'
        },
        'bacterial_leaf_blight': {
            'crops': ['rice'],
            'temp_range': (25, 35), 'humidity_min': 75,
            'risk_formula': 'quadratic'
        },
        'anthracnose': {
            'crops': ['mango', 'banana', 'chili'],
            'temp_range': (20, 30), 'humidity_min': 80,
            'rain_trigger': 5, 'risk_formula': 'exponential'
        }
    }
    
    # Base Yields (kg/hectare under optimal conditions)
    BASE_YIELDS = {
        'rice': 6000, 'wheat': 5000, 'maize': 8000, 'cotton': 2500,
        'sugarcane': 80000, 'potato': 25000, 'tomato': 40000, 'onion': 20000,
        'groundnut': 2500, 'soybean': 3000, 'mustard': 1500, 'chickpea': 2000,
        'lentil': 1500, 'jute': 3000, 'tea': 2000, 'coffee': 1500,
        'pepper': 3000, 'cardamom': 200, 'turmeric': 25000, 'banana': 40000,
        'mango': 10000, 'coconut': 15000, 'millets': 2500
    }
    
    # Soil Health Index Weights
    SOIL_HEALTH_WEIGHTS = {
        'organic_carbon': 0.25,
        'nitrogen': 0.18,
        'phosphorus': 0.12,
        'potassium': 0.12,
        'ph_balance': 0.15,
        'ec': 0.08,  # Electrical Conductivity
        'texture': 0.10
    }
    
    # Optimal Soil Parameters
    OPTIMAL_SOIL = {
        'ph': {'min': 6.0, 'max': 7.5, 'ideal': 6.5},
        'organic_carbon': {'min': 0.5, 'max': 2.0, 'ideal': 1.0},  # %
        'nitrogen': {'min': 250, 'max': 560, 'ideal': 400},  # kg/ha
        'phosphorus': {'min': 20, 'max': 60, 'ideal': 40},  # kg/ha
        'potassium': {'min': 120, 'max': 300, 'ideal': 200},  # kg/ha
        'ec': {'min': 0.2, 'max': 1.0, 'ideal': 0.5}  # dS/m
    }
    
    SOIL_TYPES = [
        'alluvial', 'black', 'red', 'laterite', 
        'arid', 'forest', 'saline', 'peaty'
    ]
    
    INDIAN_STATES = [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
        'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
        'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
        'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
        'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
    ]
    
    # Crop growing seasons (month ranges)
    CROP_SEASONS = {
        'rice': {'kharif': (6, 10), 'rabi': (11, 4)},
        'wheat': {'rabi': (10, 3)},
        'maize': {'kharif': (6, 9), 'rabi': (10, 2)},
        'cotton': {'kharif': (4, 12)},
        'sugarcane': {'year_round': (1, 12)},
        'groundnut': {'kharif': (6, 10), 'rabi': (1, 5)},
        'soybean': {'kharif': (6, 10)},
        'chickpea': {'rabi': (10, 3)},
        'mustard': {'rabi': (10, 3)},
        'onion': {'kharif': (6, 10), 'rabi': (10, 3)},
        'tomato': {'year_round': (1, 12)},
        'potato': {'rabi': (10, 3)},
    }
    
    # Water requirements (mm per season)
    CROP_WATER_NEEDS = {
        'rice': 1200, 'wheat': 450, 'maize': 500, 'cotton': 700,
        'sugarcane': 2000, 'groundnut': 500, 'soybean': 450,
        'chickpea': 300, 'mustard': 350, 'onion': 400,
        'tomato': 600, 'potato': 500, 'millets': 350,
    }
    
    # NPK requirements (kg/hectare)
    CROP_FERTILIZER = {
        'rice': {'N': 120, 'P': 60, 'K': 40},
        'wheat': {'N': 120, 'P': 60, 'K': 40},
        'maize': {'N': 150, 'P': 75, 'K': 40},
        'cotton': {'N': 150, 'P': 60, 'K': 60},
        'sugarcane': {'N': 250, 'P': 100, 'K': 100},
        'groundnut': {'N': 25, 'P': 50, 'K': 45},
        'soybean': {'N': 30, 'P': 60, 'K': 30},
        'potato': {'N': 180, 'P': 100, 'K': 150},
        'onion': {'N': 100, 'P': 50, 'K': 50},
        'tomato': {'N': 150, 'P': 100, 'K': 125},
    }
    
    # Market Prices (₹ per quintal) - Fallback/Reference
    FALLBACK_CROP_PRICES = {
        'rice': 2040,  # MSP 2023-24
        'wheat': 2275,  # MSP 2023-24
        'maize': 2090,
        'cotton': 6620,
        'sugarcane': 315,  # Per quintal
        'groundnut': 5850,
        'soybean': 4600,
        'chickpea': 5440,
        'mustard': 5650,
        'onion': 1500,
        'tomato': 1200,
        'potato': 1000,
        'turmeric': 7500,
        'banana': 2000,
        'mango': 3500,
        'coconut': 2800,
        'millets': 2500,
        'lentil': 6425,
        'jute': 5050
    }
    
    # Government schemes
    GOV_SCHEMES = [
        {
            'id': 'pmksy',
            'name': 'PM Krishi Sinchai Yojana',
            'name_hi': 'पीएम कृषि सिंचाई योजना',
            'description': 'Irrigation support and water conservation',
            'description_hi': 'सिंचाई सहायता और जल संरक्षण',
            'url': 'https://pmksy.gov.in/'
        },
        {
            'id': 'pmfby',
            'name': 'PM Fasal Bima Yojana',
            'name_hi': 'पीएम फसल बीमा योजना',
            'description': 'Crop insurance scheme',
            'description_hi': 'फसल बीमा योजना',
            'url': 'https://pmfby.gov.in/'
        },
        {
            'id': 'pmkisan',
            'name': 'PM-KISAN',
            'name_hi': 'पीएम-किसान',
            'description': 'Direct income support of ₹6000/year',
            'description_hi': '₹6000/वर्ष की प्रत्यक्ष आय सहायता',
            'url': 'https://pmkisan.gov.in/'
        },
        {
            'id': 'kcc',
            'name': 'Kisan Credit Card',
            'name_hi': 'किसान क्रेडिट कार्ड',
            'description': 'Easy credit access for farmers',
            'description_hi': 'किसानों के लिए आसान ऋण',
            'url': 'https://www.pmkisan.gov.in/KCC.aspx'
        },
        {
            'id': 'soil_health',
            'name': 'Soil Health Card',
            'name_hi': 'मृदा स्वास्थ्य कार्ड',
            'description': 'Free soil testing and recommendations',
            'description_hi': 'मुफ्त मिट्टी परीक्षण और सिफारिशें',
            'url': 'https://soilhealth.dac.gov.in/'
        },
        {
            'id': 'enam',
            'name': 'e-NAM',
            'name_hi': 'ई-नाम',
            'description': 'Online trading platform for agricultural commodities',
            'description_hi': 'कृषि वस्तुओं के लिए ऑनलाइन व्यापार मंच',
            'url': 'https://enam.gov.in/'
        },
    ]
    
    # Pest alerts based on conditions
    PEST_CONDITIONS = {
        'high_humidity_high_temp': {
            'pests': ['aphids', 'whitefly', 'thrips'],
            'crops_affected': ['cotton', 'tomato', 'chili'],
            'alert_level': 'high'
        },
        'high_humidity_moderate_temp': {
            'pests': ['fungal_diseases', 'leaf_blight', 'rust'],
            'crops_affected': ['wheat', 'rice', 'potato'],
            'alert_level': 'medium'
        },
        'low_humidity_high_temp': {
            'pests': ['spider_mites', 'grasshoppers'],
            'crops_affected': ['groundnut', 'millets'],
            'alert_level': 'medium'
        }
    }
