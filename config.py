import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'ceres-dtl-secret-key-change-in-production')
    
    # FREE APIs - No keys required
    OPEN_METEO_BASE = "https://api.open-meteo.com/v1"
    NOMINATIM_BASE = "https://nominatim.openstreetmap.org"
    
    # Optional APIs (free tiers available)
    OPENWEATHERMAP_API_KEY = os.environ.get('OPENWEATHERMAP_API_KEY', '')
    DATA_GOV_API_KEY = os.environ.get('DATA_GOV_API_KEY', '')
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '')
    
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///ceres.db')
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'crop_model.pkl')
    
    SUPPORTED_LANGUAGES = ['en', 'hi', 'kn', 'te', 'ta', 'mr']
    DEFAULT_LANGUAGE = 'en'
    
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
