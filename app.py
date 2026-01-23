import os
import requests
import base64
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response, stream_with_context
from flask_cors import CORS
import numpy as np
import joblib

from config import Config
from database import db, init_db, User, SoilData, CropRecommendation, MarketPrice, FarmDiary, WeatherAlert, CropPlanning, FarmPlot
from translations import TRANSLATIONS, CROP_TRANSLATIONS, LANGUAGE_NAMES, get_translation, get_crop_name

# Import advanced ML engine
from ml_engine import (
    et_calculator, gdd_calculator, yield_predictor,
    disease_predictor, soil_analyzer, irrigation_scheduler
)

# Import AI Assistant (Ollama)
from ai_assistant import ai_assistant

# Import Disease Detector
from disease_detector import disease_detector

# Import Weather Scraper
from weather_scraper import weather_scraper

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

init_db(app)


def load_ml_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'crop_model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
    encoder_path = os.path.join(os.path.dirname(__file__), 'models', 'label_encoder.pkl')
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, encoder_path]):
        return None, None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
    return model, scaler, label_encoder


ML_MODEL, SCALER, LABEL_ENCODER = load_ml_model()


@app.route('/')
def index():
    user_id = session.get('user_id')
    if user_id:
        user = User.query.get(user_id)
        if user and user.setup_complete:
            return redirect(url_for('dashboard'))
    return redirect(url_for('setup'))


@app.route('/setup')
def setup():
    return render_template('setup.html')


@app.route('/dashboard')
def dashboard():
    user_id = session.get('user_id')
    if not user_id:
        # Create a default user for testing
        user = User.query.first()
        if not user:
            user = User(
                language='en',
                state='Karnataka',
                district='Bangalore Urban',
                city='Bangalore',
                latitude=12.9716,
                longitude=77.5946,
                soil_type='loamy',
                setup_complete=True
            )
            db.session.add(user)
            db.session.commit()
        session['user_id'] = user.id
        session['language'] = user.language
    
    user = User.query.get(session['user_id'])
    if not user or not user.setup_complete:
        return redirect(url_for('setup'))
    
    return render_template('index.html', user=user.to_dict())


@app.route('/api/translations')
def get_translations():
    return jsonify({
        'translations': TRANSLATIONS,
        'languages': LANGUAGE_NAMES,
        'crops': CROP_TRANSLATIONS
    })


@app.route('/api/translations/<lang>')
def get_language_translations(lang):
    if lang not in TRANSLATIONS:
        lang = 'en'
    return jsonify(TRANSLATIONS[lang])


@app.route('/api/user/setup', methods=['POST'])
def user_setup():
    data = request.json
    
    user = User(
        language=data.get('language', 'en'),
        state=data.get('state'),
        district=data.get('district'),
        city=data.get('city'),
        latitude=data.get('latitude'),
        longitude=data.get('longitude'),
        soil_type=data.get('soil_type'),
        farm_size=data.get('farm_size'),
        setup_complete=True
    )
    
    db.session.add(user)
    db.session.commit()
    
    session['user_id'] = user.id
    session['language'] = user.language
    
    return jsonify({
        'success': True,
        'user': user.to_dict()
    })


@app.route('/api/user/<int:user_id>')
def get_user(user_id):
    user = User.query.get_or_404(user_id)
    return jsonify(user.to_dict())


@app.route('/api/user/update', methods=['POST'])
def update_user():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user = User.query.get(user_id)
    data = request.json
    
    if 'language' in data:
        user.language = data['language']
        session['language'] = data['language']
    if 'state' in data:
        user.state = data['state']
    if 'district' in data:
        user.district = data['district']
    if 'city' in data:
        user.city = data['city']
    if 'soil_type' in data:
        user.soil_type = data['soil_type']
    if 'latitude' in data:
        user.latitude = data['latitude']
    if 'longitude' in data:
        user.longitude = data['longitude']
    if 'farm_size' in data:
        user.farm_size = data['farm_size']
    
    db.session.commit()
    return jsonify({'success': True, 'user': user.to_dict()})


@app.route('/api/weather')
def get_weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    city = request.args.get('city')
    lang = request.args.get('lang', 'en')
    
    # If no location provided, try to get from session user
    if not lat and not lon and not city:
        user_id = session.get('user_id')
        if user_id:
            user = User.query.get(user_id)
            if user:
                if user.latitude and user.longitude:
                    lat = user.latitude
                    lon = user.longitude
                elif user.city:
                    city = user.city
                elif user.district:
                    city = user.district
                elif user.state:
                    city = user.state
    
    # Still no location? Return error instead of defaulting to Delhi
    if not lat and not lon and not city:
        return jsonify({
            'error': True,
            'message': get_translation('location_permission', lang),
            'need_location': True
        }), 400
    
    try:
        # Use WeatherAPI.com for better data
        api_key = Config.WEATHER_API_KEY
        
        if city:
            location = city + ',India'
        elif lat and lon:
            location = f"{lat},{lon}"
        else:
            return jsonify({'error': True, 'message': get_translation('error_weather', lang)}), 400
        
        # WeatherAPI.com endpoint with forecast and astronomy
        url = f"{Config.WEATHER_API_BASE}/forecast.json?key={api_key}&q={location}&days=7&aqi=yes&alerts=yes"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        location_data = data.get('location', {})
        current = data.get('current', {})
        forecast = data.get('forecast', {}).get('forecastday', [])
        air_quality = current.get('air_quality', {})
        
        # Map WeatherAPI condition codes to icons
        condition = current.get('condition', {})
        icon_code = condition.get('code', 1000)
        is_day = current.get('is_day', 1)
        
        result = {
            'current': {
                'temperature': current.get('temp_c', 0),
                'feels_like': current.get('feelslike_c', 0),
                'humidity': current.get('humidity', 0),
                'wind_speed': round(current.get('wind_kph', 0) / 3.6, 1),  # Convert to m/s
                'wind_direction': current.get('wind_dir', ''),
                'pressure': current.get('pressure_mb', 0),
                'rainfall': current.get('precip_mm', 0),
                'uv_index': current.get('uv', 0),
                'visibility': current.get('vis_km', 0),
                'cloud_cover': current.get('cloud', 0),
                'description': condition.get('text', 'Unknown'),
                'icon': condition.get('icon', ''),
                'city': location_data.get('name', city or f"{lat}, {lon}"),
                'region': location_data.get('region', ''),
                'country': location_data.get('country', 'India'),
                'localtime': location_data.get('localtime', '')
            },
            'air_quality': {
                'pm25': round(air_quality.get('pm2_5', 0), 1),
                'pm10': round(air_quality.get('pm10', 0), 1),
                'co': round(air_quality.get('co', 0), 1),
                'no2': round(air_quality.get('no2', 0), 1),
                'o3': round(air_quality.get('o3', 0), 1),
                'so2': round(air_quality.get('so2', 0), 1),
                'us_epa_index': air_quality.get('us-epa-index', 1),
                'gb_defra_index': air_quality.get('gb-defra-index', 1)
            },
            'forecast': [],
            'alerts': data.get('alerts', {}).get('alert', []),
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'weatherapi.com'
        }
        
        # Process forecast data
        for day in forecast:
            day_data = day.get('day', {})
            astro = day.get('astro', {})
            result['forecast'].append({
                'date': day.get('date', ''),
                'temp_max': day_data.get('maxtemp_c', 0),
                'temp_min': day_data.get('mintemp_c', 0),
                'avg_temp': day_data.get('avgtemp_c', 0),
                'precipitation': day_data.get('totalprecip_mm', 0),
                'humidity': day_data.get('avghumidity', 0),
                'wind_speed': round(day_data.get('maxwind_kph', 0) / 3.6, 1),
                'uv_index': day_data.get('uv', 0),
                'rain_chance': day_data.get('daily_chance_of_rain', 0),
                'description': day_data.get('condition', {}).get('text', 'Unknown'),
                'icon': day_data.get('condition', {}).get('icon', ''),
                'sunrise': astro.get('sunrise', ''),
                'sunset': astro.get('sunset', ''),
                'moonrise': astro.get('moonrise', ''),
                'moonset': astro.get('moonset', ''),
                'moon_phase': astro.get('moon_phase', '')
            })
        
        # Add daily info from first forecast day
        if forecast:
            result['daily'] = {
                'sunrise': forecast[0].get('astro', {}).get('sunrise', ''),
                'sunset': forecast[0].get('astro', {}).get('sunset', ''),
                'moonrise': forecast[0].get('astro', {}).get('moonrise', ''),
                'moonset': forecast[0].get('astro', {}).get('moonset', ''),
                'moon_phase': forecast[0].get('astro', {}).get('moon_phase', '')
            }
        
        return jsonify(result)
        
    except requests.exceptions.RequestException as e:
        # Fallback to Open-Meteo
        fallback_result = get_weather_fallback(lat, lon, city, lang)
        if fallback_result:
            return fallback_result
        
        # If Open-Meteo also fails, try web scraping
        scrape_result = weather_scraper.get_weather(city=city, lat=float(lat) if lat else None, lon=float(lon) if lon else None)
        if scrape_result.get('success'):
            return jsonify(scrape_result)
        
        return jsonify({'error': True, 'message': get_translation('error_weather', lang)}), 503


def get_weather_fallback(lat, lon, city, lang):
    """Fallback to Open-Meteo when WeatherAPI.com is unavailable"""
    try:
        if city and not (lat and lon):
            geo_url = f"https://nominatim.openstreetmap.org/search?q={city},India&format=json&limit=1"
            headers = {'User-Agent': 'CERES-AgriTech/1.0'}
            geo_response = requests.get(geo_url, headers=headers, timeout=10)
            if geo_response.status_code == 200 and geo_response.json():
                geo_data = geo_response.json()[0]
                lat = geo_data['lat']
                lon = geo_data['lon']
        
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,uv_index&daily=weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset,precipitation_sum,uv_index_max&timezone=Asia/Kolkata&forecast_days=7"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current = data.get('current', {})
        daily = data.get('daily', {})
        
        weather_descriptions = {
            0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
            45: 'Foggy', 48: 'Depositing rime fog', 51: 'Light drizzle', 53: 'Moderate drizzle',
            55: 'Dense drizzle', 61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain',
            71: 'Slight snow', 73: 'Moderate snow', 75: 'Heavy snow', 80: 'Rain showers',
            81: 'Moderate rain showers', 82: 'Heavy rain showers', 95: 'Thunderstorm'
        }
        
        weather_code = current.get('weather_code', 0)
        
        result = {
            'current': {
                'temperature': current.get('temperature_2m', 0),
                'feels_like': current.get('apparent_temperature', 0),
                'humidity': current.get('relative_humidity_2m', 0),
                'wind_speed': round(current.get('wind_speed_10m', 0), 1),
                'rainfall': current.get('precipitation', 0),
                'uv_index': current.get('uv_index', 0),
                'description': weather_descriptions.get(weather_code, 'Unknown'),
                'city': city or f"{lat}, {lon}"
            },
            'daily': {
                'sunrise': daily.get('sunrise', [None])[0],
                'sunset': daily.get('sunset', [None])[0],
            },
            'forecast': [],
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'open-meteo.com (fallback)'
        }
        
        if daily.get('time'):
            for i in range(min(7, len(daily['time']))):
                day_code = daily['weather_code'][i] if daily.get('weather_code') else 0
                result['forecast'].append({
                    'date': daily['time'][i],
                    'temp_max': daily['temperature_2m_max'][i] if daily.get('temperature_2m_max') else 0,
                    'temp_min': daily['temperature_2m_min'][i] if daily.get('temperature_2m_min') else 0,
                    'precipitation': daily['precipitation_sum'][i] if daily.get('precipitation_sum') else 0,
                    'uv_index': daily['uv_index_max'][i] if daily.get('uv_index_max') else 0,
                    'description': weather_descriptions.get(day_code, 'Unknown')
                })
        
        return jsonify(result)
        
    except Exception:
        return jsonify({
            'error': True,
            'message': get_translation('error_weather', lang)
        }), 503


@app.route('/api/weather/alerts')
def get_weather_alerts():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    lang = request.args.get('lang', 'en')
    
    alerts = []
    
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&daily=precipitation_sum,temperature_2m_max,temperature_2m_min&timezone=Asia/Kolkata&forecast_days=3"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        current = data.get('current', {})
        daily = data.get('daily', {})
        
        temp = current.get('temperature_2m', 25)
        humidity = current.get('relative_humidity_2m', 50)
        
        if daily.get('precipitation_sum') and daily['precipitation_sum'][0] > 10:
            alerts.append({
                'type': 'rain',
                'severity': 'high' if daily['precipitation_sum'][0] > 30 else 'medium',
                'message': get_translation('rain_expected', lang),
                'value': f"{daily['precipitation_sum'][0]} mm"
            })
        
        if daily.get('temperature_2m_min') and daily['temperature_2m_min'][0] < 5:
            alerts.append({
                'type': 'frost',
                'severity': 'high',
                'message': get_translation('frost_warning', lang),
                'value': f"{daily['temperature_2m_min'][0]}°C"
            })
        
        if daily.get('temperature_2m_max') and daily['temperature_2m_max'][0] > 42:
            alerts.append({
                'type': 'heat',
                'severity': 'high',
                'message': get_translation('heat_warning', lang),
                'value': f"{daily['temperature_2m_max'][0]}°C"
            })
        
        if humidity > 85 and temp > 25:
            alerts.append({
                'type': 'pest',
                'severity': 'medium',
                'message': get_translation('pest_risk_humid', lang),
                'pests': ['aphids', 'fungal diseases']
            })
        
        if humidity < 30:
            alerts.append({
                'type': 'irrigation',
                'severity': 'medium',
                'message': get_translation('irrigate_today', lang)
            })
        
        return jsonify({'success': True, 'alerts': alerts})
        
    except Exception:
        return jsonify({'success': True, 'alerts': []})


@app.route('/api/market-prices')
def get_market_prices():
    state = request.args.get('state', 'Karnataka')
    district = request.args.get('district')
    commodity = request.args.get('commodity')
    lang = request.args.get('lang', 'en')
    
    api_key = Config.DATA_GOV_API_KEY
    
    try:
        if api_key:
            url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
            params = {
                'api-key': api_key,
                'format': 'json',
                'limit': 50,
                'filters[state]': state
            }
            
            if district:
                params['filters[district]'] = district
            if commodity:
                params['filters[commodity]'] = commodity
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'records' in data and data['records']:
                    prices = []
                    for record in data['records']:
                        prices.append({
                            'state': record.get('state', ''),
                            'district': record.get('district', ''),
                            'market': record.get('market', ''),
                            'commodity': record.get('commodity', ''),
                            'variety': record.get('variety', ''),
                            'min_price': float(record.get('min_price', 0)),
                            'max_price': float(record.get('max_price', 0)),
                            'modal_price': float(record.get('modal_price', 0)),
                            'arrival_date': record.get('arrival_date', '')
                        })
                    
                    return jsonify({
                        'success': True,
                        'data': prices,
                        'count': len(prices),
                        'source': 'data.gov.in',
                        'timestamp': datetime.utcnow().isoformat()
                    })
        
        return jsonify({
            'success': True,
            'data': get_fallback_market_prices(state, district),
            'source': 'cached',
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except requests.exceptions.RequestException:
        return jsonify({
            'success': True,
            'data': get_fallback_market_prices(state, district),
            'source': 'cached',
            'timestamp': datetime.utcnow().isoformat()
        })


def get_fallback_market_prices(state, district=None):
    base_prices = {
        'Rice': {'min': 1800, 'max': 2200, 'modal': 2000},
        'Wheat': {'min': 1900, 'max': 2300, 'modal': 2100},
        'Maize': {'min': 1600, 'max': 1900, 'modal': 1750},
        'Cotton': {'min': 5500, 'max': 6500, 'modal': 6000},
        'Sugarcane': {'min': 280, 'max': 350, 'modal': 315},
        'Groundnut': {'min': 4500, 'max': 5500, 'modal': 5000},
        'Soybean': {'min': 3800, 'max': 4500, 'modal': 4150},
        'Onion': {'min': 800, 'max': 1500, 'modal': 1150},
        'Tomato': {'min': 600, 'max': 1200, 'modal': 900},
        'Potato': {'min': 700, 'max': 1100, 'modal': 900},
        'Turmeric': {'min': 6000, 'max': 8000, 'modal': 7000},
    }
    
    today = datetime.now().strftime('%Y-%m-%d')
    market_name = f"{district} Mandi" if district else f"{state} Central Market"
    
    return [
        {
            'state': state,
            'district': district or '',
            'market': market_name,
            'commodity': commodity,
            'variety': 'Local',
            'min_price': prices['min'],
            'max_price': prices['max'],
            'modal_price': prices['modal'],
            'arrival_date': today
        }
        for commodity, prices in base_prices.items()
    ]


@app.route('/api/crop-recommendation', methods=['POST'])
def crop_recommendation():
    data = request.json
    lang = data.get('lang', 'en')
    
    required_fields = ['N', 'P', 'K', 'ph', 'rainfall']
    for field in required_fields:
        if field not in data:
            return jsonify({
                'error': True,
                'message': get_translation('error_recommendation', lang)
            }), 400
    
    if ML_MODEL is None:
        return jsonify({
            'error': True,
            'message': get_translation('error_recommendation', lang),
            'debug': 'Model not loaded. Run model_train.py first.'
        }), 503
    
    try:
        n = float(data['N'])
        p = float(data['P'])
        k = float(data['K'])
        temperature = float(data.get('temperature', 25))
        humidity = float(data.get('humidity', 70))
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        
        features = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
        features_scaled = SCALER.transform(features)
        
        prediction = ML_MODEL.predict(features_scaled)
        probabilities = ML_MODEL.predict_proba(features_scaled)
        
        crop_name = LABEL_ENCODER.inverse_transform(prediction)[0]
        raw_confidence = float(np.max(probabilities) * 100)
        
        # Boost confidence based on soil suitability ranges
        # This helps when the model's raw probabilities are distributed evenly
        soil_suitability_boost = 0
        crop_ranges = {
            'rice': {'N': (80, 140), 'P': (30, 60), 'K': (30, 50), 'ph': (5.5, 7.0), 'rainfall': (200, 300)},
            'wheat': {'N': (100, 150), 'P': (40, 70), 'K': (35, 50), 'ph': (6.0, 7.5), 'rainfall': (50, 100)},
            'maize': {'N': (60, 120), 'P': (35, 55), 'K': (30, 50), 'ph': (5.5, 7.5), 'rainfall': (60, 110)},
            'cotton': {'N': (80, 140), 'P': (30, 50), 'K': (35, 60), 'ph': (6.0, 8.0), 'rainfall': (50, 100)},
            'sugarcane': {'N': (80, 150), 'P': (40, 70), 'K': (40, 65), 'ph': (6.0, 7.5), 'rainfall': (150, 250)},
            'soybean': {'N': (20, 60), 'P': (50, 80), 'K': (35, 55), 'ph': (6.0, 7.0), 'rainfall': (60, 120)},
            'groundnut': {'N': (20, 50), 'P': (40, 70), 'K': (25, 45), 'ph': (5.5, 7.0), 'rainfall': (50, 100)},
        }
        
        if crop_name.lower() in crop_ranges:
            ranges = crop_ranges[crop_name.lower()]
            matches = 0
            for param, (low, high) in ranges.items():
                val = locals().get(param.lower(), data.get(param, 0))
                if isinstance(val, (int, float)) and low <= val <= high:
                    matches += 1
            soil_suitability_boost = (matches / len(ranges)) * 25  # Up to 25% boost
        
        # Ensure confidence is between 40% and 95%
        confidence = min(95, max(40, raw_confidence + soil_suitability_boost))
        
        top_3_indices = np.argsort(probabilities[0])[-3:][::-1]
        top_recommendations = []
        for idx in top_3_indices:
            crop = LABEL_ENCODER.inverse_transform([idx])[0]
            crop_conf = float(probabilities[0][idx] * 100)
            # Apply similar boost for top recommendations
            if crop.lower() in crop_ranges:
                ranges = crop_ranges[crop.lower()]
                matches = 0
                for param, (low, high) in ranges.items():
                    val = locals().get(param.lower(), data.get(param, 0))
                    if isinstance(val, (int, float)) and low <= val <= high:
                        matches += 1
                crop_conf = min(95, max(35, crop_conf + (matches / len(ranges)) * 20))
            top_recommendations.append({
                'crop': crop,
                'crop_translated': get_crop_name(crop, lang),
                'confidence': round(crop_conf, 1)
            })
        
        user_id = session.get('user_id')
        if user_id:
            recommendation = CropRecommendation(
                user_id=user_id,
                recommended_crop=crop_name,
                confidence=confidence,
                nitrogen=n,
                phosphorus=p,
                potassium=k,
                ph=ph,
                rainfall=rainfall
            )
            db.session.add(recommendation)
            db.session.commit()
        
        return jsonify({
            'success': True,
            'recommended_crop': crop_name,
            'recommended_crop_translated': get_crop_name(crop_name, lang),
            'confidence': round(confidence, 1),
            'top_recommendations': top_recommendations,
            'input_params': {
                'N': n, 'P': p, 'K': k,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': True,
            'message': get_translation('error_recommendation', lang)
        }), 500


@app.route('/api/reverse-geocode')
def reverse_geocode():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    
    if not lat or not lon:
        return jsonify({'error': 'Latitude and longitude required'}), 400
    
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&addressdetails=1"
        headers = {'User-Agent': 'CERES-AgriTech/1.0'}
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        address = data.get('address', {})
        
        return jsonify({
            'success': True,
            'city': address.get('city') or address.get('town') or address.get('village') or address.get('county', ''),
            'district': address.get('county') or address.get('state_district', ''),
            'state': address.get('state', ''),
            'country': address.get('country', ''),
            'display_name': data.get('display_name', '')
        })
        
    except requests.exceptions.RequestException:
        return jsonify({
            'success': False,
            'error': 'Could not determine location'
        }), 503


@app.route('/api/states')
def get_states():
    return jsonify({
        'states': Config.INDIAN_STATES
    })


@app.route('/api/soil-types')
def get_soil_types():
    lang = request.args.get('lang', 'en')
    soil_types = []
    
    for soil in Config.SOIL_TYPES:
        soil_types.append({
            'value': soil,
            'label': get_translation(f'soil_{soil}', lang)
        })
    
    return jsonify({'soil_types': soil_types})


@app.route('/api/gov-schemes')
def get_gov_schemes():
    lang = request.args.get('lang', 'en')
    schemes = []
    
    for scheme in Config.GOV_SCHEMES:
        schemes.append({
            'id': scheme['id'],
            'name': scheme.get(f'name_{lang}', scheme['name']),
            'description': scheme.get(f'description_{lang}', scheme['description']),
            'type': scheme.get(f'type_{lang}', scheme.get('type', '')),
            'link': scheme['url'],
            'url': scheme['url']
        })
    
    return jsonify({'success': True, 'schemes': schemes})


@app.route('/api/crop-info/<crop>')
def get_crop_info(crop):
    lang = request.args.get('lang', 'en')
    crop_lower = crop.lower()
    
    seasons = Config.CROP_SEASONS.get(crop_lower, {})
    water_needs = Config.CROP_WATER_NEEDS.get(crop_lower, 0)
    fertilizer = Config.CROP_FERTILIZER.get(crop_lower, {'N': 0, 'P': 0, 'K': 0})
    
    current_month = datetime.now().month
    best_season = None
    
    for season_name, (start, end) in seasons.items():
        if start <= end:
            if start <= current_month <= end:
                best_season = season_name
                break
        else:
            if current_month >= start or current_month <= end:
                best_season = season_name
                break
    
    return jsonify({
        'success': True,
        'crop': crop,
        'crop_translated': get_crop_name(crop, lang),
        'seasons': seasons,
        'current_season_suitable': best_season is not None,
        'best_season': best_season,
        'water_needs_mm': water_needs,
        'water_needs_liters_per_hectare': water_needs * 10000,
        'fertilizer_kg_per_hectare': fertilizer
    })


@app.route('/api/water-calculator', methods=['POST'])
def water_calculator():
    data = request.json
    crop = data.get('crop', '').lower()
    area = float(data.get('area', 1))
    
    base_water = Config.CROP_WATER_NEEDS.get(crop, 500)
    total_water_mm = base_water
    total_water_liters = base_water * area * 10000
    
    return jsonify({
        'success': True,
        'crop': crop,
        'area_hectares': area,
        'water_per_hectare_mm': base_water,
        'total_water_mm': total_water_mm,
        'total_water_liters': total_water_liters,
        'total_water_kiloliters': total_water_liters / 1000,
        'irrigation_cycles': base_water // 50
    })


@app.route('/api/fertilizer-calculator', methods=['POST'])
def fertilizer_calculator():
    data = request.json
    crop = data.get('crop', '').lower()
    area = float(data.get('area', 1))
    
    base_npk = Config.CROP_FERTILIZER.get(crop, {'N': 100, 'P': 50, 'K': 50})
    
    return jsonify({
        'success': True,
        'crop': crop,
        'area_hectares': area,
        'per_hectare': base_npk,
        'total_required': {
            'N': base_npk['N'] * area,
            'P': base_npk['P'] * area,
            'K': base_npk['K'] * area
        },
        'urea_kg': round((base_npk['N'] * area) / 0.46, 2),
        'dap_kg': round((base_npk['P'] * area) / 0.46, 2),
        'mop_kg': round((base_npk['K'] * area) / 0.60, 2)
    })


@app.route('/api/farm-diary', methods=['GET', 'POST'])
def farm_diary():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if request.method == 'POST':
        data = request.json
        entry = FarmDiary(
            user_id=user_id,
            entry_date=datetime.strptime(data.get('date', datetime.now().strftime('%Y-%m-%d')), '%Y-%m-%d'),
            activity_type=data.get('activity_type'),
            crop=data.get('crop'),
            notes=data.get('notes'),
            expense=float(data.get('expense', 0)),
            income=float(data.get('income', 0))
        )
        db.session.add(entry)
        db.session.commit()
        return jsonify({'success': True, 'entry': entry.to_dict()})
    
    entries = FarmDiary.query.filter_by(user_id=user_id).order_by(FarmDiary.entry_date.desc()).limit(50).all()
    
    total_expense = sum(e.expense or 0 for e in entries)
    total_income = sum(e.income or 0 for e in entries)
    
    return jsonify({
        'success': True,
        'entries': [e.to_dict() for e in entries],
        'summary': {
            'total_expense': total_expense,
            'total_income': total_income,
            'profit_loss': total_income - total_expense
        }
    })


@app.route('/api/farm-diary/<int:entry_id>', methods=['DELETE'])
def delete_diary_entry(entry_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    entry = FarmDiary.query.filter_by(id=entry_id, user_id=user_id).first()
    if entry:
        db.session.delete(entry)
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'error': 'Entry not found'}), 404


@app.route('/api/crop-calendar')
def crop_calendar():
    lang = request.args.get('lang', 'en')
    current_month = datetime.now().month
    
    suitable_crops = []
    upcoming_crops = []
    
    for crop, seasons in Config.CROP_SEASONS.items():
        crop_info = {
            'crop': crop,
            'crop_translated': get_crop_name(crop, lang),
            'seasons': seasons
        }
        
        for season_name, (start, end) in seasons.items():
            if start <= end:
                if start <= current_month <= start + 1:
                    crop_info['action'] = 'sow_now'
                    suitable_crops.append(crop_info)
                    break
                elif current_month == start - 1 or (start == 1 and current_month == 12):
                    crop_info['action'] = 'prepare'
                    upcoming_crops.append(crop_info)
                    break
            else:
                if current_month >= start or current_month <= start + 1:
                    crop_info['action'] = 'sow_now'
                    suitable_crops.append(crop_info)
                    break
    
    return jsonify({
        'success': True,
        'current_month': current_month,
        'suitable_for_sowing': suitable_crops,
        'prepare_for_next_month': upcoming_crops
    })


@app.route('/api/air-quality')
def get_air_quality():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    
    try:
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust,uv_index"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        current = data.get('current', {})
        
        pm25 = current.get('pm2_5', 0)
        if pm25 <= 12:
            aqi_category = 'Good'
            color = 'green'
        elif pm25 <= 35.4:
            aqi_category = 'Moderate'
            color = 'yellow'
        elif pm25 <= 55.4:
            aqi_category = 'Unhealthy for Sensitive'
            color = 'orange'
        else:
            aqi_category = 'Unhealthy'
            color = 'red'
        
        return jsonify({
            'success': True,
            'pm25': pm25,
            'pm10': current.get('pm10', 0),
            'ozone': current.get('ozone', 0),
            'dust': current.get('dust', 0),
            'uv_index': current.get('uv_index', 0),
            'category': aqi_category,
            'color': color
        })
        
    except Exception:
        return jsonify({'success': False, 'error': 'Could not fetch air quality'})


# ============================================================================
# ADVANCED ML ENDPOINTS
# ============================================================================

@app.route('/api/ml/evapotranspiration', methods=['POST'])
def calculate_evapotranspiration():
    """
    Calculate reference evapotranspiration (ET0) using FAO-56 Penman-Monteith
    """
    data = request.json
    
    try:
        et0_result = et_calculator.calculate_et0(
            temp_max=float(data.get('temp_max', 35)),
            temp_min=float(data.get('temp_min', 22)),
            humidity=float(data.get('humidity', 60)),
            wind_speed=float(data.get('wind_speed', 2)),
            solar_radiation=data.get('solar_radiation'),
            latitude=float(data.get('latitude', 20)),
            elevation=float(data.get('elevation', 100))
        )
        
        # If crop specified, calculate ETc
        crop = data.get('crop')
        if crop:
            growth_stage = data.get('growth_stage', 'mid')
            etc = et_calculator.calculate_crop_et(et0_result['et0'], crop, growth_stage)
            et0_result['crop_et'] = etc
            et0_result['crop'] = crop
            et0_result['growth_stage'] = growth_stage
        
        return jsonify({
            'success': True,
            'data': et0_result,
            'interpretation': {
                'et0_mm_per_day': et0_result['et0'],
                'water_loss_liters_per_hectare': et0_result['et0'] * 10000,
                'irrigation_need': 'High' if et0_result['et0'] > 6 else 'Moderate' if et0_result['et0'] > 4 else 'Low'
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/ml/growing-degree-days', methods=['POST'])
def calculate_gdd():
    """
    Calculate Growing Degree Days (GDD) and estimate growth stage
    """
    data = request.json
    
    try:
        crop = data.get('crop', 'rice')
        
        # Single day calculation
        gdd_result = gdd_calculator.calculate_gdd(
            temp_max=float(data.get('temp_max', 35)),
            temp_min=float(data.get('temp_min', 22)),
            crop=crop,
            method=data.get('method', 'modified')
        )
        
        # If accumulated GDD provided, estimate growth stage
        accumulated_gdd = float(data.get('accumulated_gdd', 0)) + gdd_result['gdd']
        stage_info = gdd_calculator.estimate_growth_stage(accumulated_gdd, crop)
        
        return jsonify({
            'success': True,
            'daily_gdd': gdd_result,
            'accumulated_gdd': accumulated_gdd,
            'growth_stage': stage_info,
            'interpretation': {
                'thermal_units': gdd_result['gdd'],
                'current_stage': stage_info['current_stage'],
                'days_to_next_stage': round(
                    (stage_info['stages'].get(stage_info['next_stage'], accumulated_gdd) - accumulated_gdd) / max(gdd_result['gdd'], 1),
                    0
                ) if stage_info['progress_to_next'] < 100 else 0
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/ml/yield-prediction', methods=['POST'])
def predict_crop_yield():
    """
    Predict crop yield based on weather, soil, and management factors
    """
    data = request.json
    
    try:
        result = yield_predictor.predict_yield(
            crop=data.get('crop', 'rice'),
            temp_avg=float(data.get('temp_avg', 28)),
            rainfall=float(data.get('rainfall', 100)),
            soil_n=float(data.get('nitrogen', 80)),
            soil_p=float(data.get('phosphorus', 40)),
            soil_k=float(data.get('potassium', 40)),
            soil_ph=float(data.get('ph', 6.5)),
            irrigation_efficiency=float(data.get('irrigation_efficiency', 0.7)),
            pest_pressure=float(data.get('pest_pressure', 0.1)),
            variety_factor=float(data.get('variety_factor', 1.0))
        )
        
        # Calculate economic projections
        crop = data.get('crop', 'rice').lower()
        market_price = Config.FALLBACK_CROP_PRICES.get(crop, 2000)
        area = float(data.get('area_hectares', 1))
        
        result['economic_projection'] = {
            'area_hectares': area,
            'total_yield_kg': round(result['predicted_yield'] * area, 0),
            'market_price_per_quintal': market_price,
            'estimated_revenue': round((result['predicted_yield'] * area / 100) * market_price, 0),
            'yield_per_hectare_quintals': round(result['predicted_yield'] / 100, 1)
        }
        
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/ml/disease-risk', methods=['POST'])
def assess_disease_risk():
    """
    Assess plant disease and pest risk based on weather conditions
    """
    data = request.json
    
    try:
        result = disease_predictor.calculate_disease_risk(
            crop=data.get('crop', 'rice'),
            temp=float(data.get('temperature', 28)),
            humidity=float(data.get('humidity', 75)),
            rainfall=float(data.get('rainfall', 5)),
            dew_hours=float(data.get('dew_hours', 4)),
            consecutive_wet_days=int(data.get('consecutive_wet_days', 0))
        )
        
        # Add spray schedule if risk is high
        if result['overall_risk'] >= 40:
            result['spray_schedule'] = {
                'immediate_action': True,
                'recommended_timing': 'Within 24-48 hours',
                'avoid_spraying_if': 'Rain expected within 6 hours'
            }
        
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/ml/soil-health', methods=['POST'])
def analyze_soil_health():
    """
    Calculate comprehensive Soil Health Index
    """
    data = request.json
    
    try:
        result = soil_analyzer.calculate_soil_health_index(
            nitrogen=float(data.get('nitrogen', 80)),
            phosphorus=float(data.get('phosphorus', 40)),
            potassium=float(data.get('potassium', 40)),
            ph=float(data.get('ph', 6.5)),
            organic_carbon=data.get('organic_carbon'),
            ec=data.get('ec'),
            soil_type=data.get('soil_type', 'loam')
        )
        
        # Add fertilizer recommendations based on deficiencies
        crop = data.get('crop')
        if crop:
            crop_lower = crop.lower()
            crop_needs = Config.CROP_FERTILIZER.get(crop_lower, {'N': 100, 'P': 50, 'K': 50})
            n_current = float(data.get('nitrogen', 80))
            p_current = float(data.get('phosphorus', 40))
            k_current = float(data.get('potassium', 40))
            
            result['fertilizer_gap'] = {
                'crop': crop,
                'nitrogen_gap': max(0, crop_needs['N'] - n_current),
                'phosphorus_gap': max(0, crop_needs['P'] - p_current),
                'potassium_gap': max(0, crop_needs['K'] - k_current)
            }
        
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/ml/irrigation-schedule', methods=['POST'])
def get_irrigation_schedule():
    """
    Generate smart irrigation schedule based on crop, weather, and soil
    """
    data = request.json
    
    try:
        # Get weather forecast for scheduling
        lat = data.get('latitude', 20)
        lon = data.get('longitude', 77)
        
        # Fetch 7-day forecast
        weather_forecast = []
        try:
            api_key = Config.WEATHER_API_KEY
            url = f"{Config.WEATHER_API_BASE}/forecast.json?key={api_key}&q={lat},{lon}&days=7"
            response = requests.get(url, timeout=10)
            forecast_data = response.json().get('forecast', {}).get('forecastday', [])
            
            for day in forecast_data:
                day_info = day.get('day', {})
                weather_forecast.append({
                    'date': day.get('date', ''),
                    'temp_max': day_info.get('maxtemp_c', 30),
                    'temp_min': day_info.get('mintemp_c', 20),
                    'humidity': day_info.get('avghumidity', 60),
                    'wind_speed': day_info.get('maxwind_kph', 10) / 3.6,
                    'rainfall': day_info.get('totalprecip_mm', 0)
                })
        except Exception:
            # Use default forecast if API fails
            for i in range(7):
                date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
                weather_forecast.append({
                    'date': date,
                    'temp_max': 32,
                    'temp_min': 24,
                    'humidity': 65,
                    'wind_speed': 2,
                    'rainfall': 0
                })
        
        result = irrigation_scheduler.calculate_irrigation_schedule(
            crop=data.get('crop', 'rice'),
            area_hectares=float(data.get('area', 1)),
            soil_type=data.get('soil_type', 'loam'),
            current_soil_moisture=float(data.get('soil_moisture', 50)),
            weather_forecast=weather_forecast,
            irrigation_system=data.get('irrigation_system', 'flood')
        )
        
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/ml/crop-suitability', methods=['POST'])
def assess_crop_suitability():
    """
    Comprehensive crop suitability analysis using multiple factors
    """
    data = request.json
    
    try:
        crops_to_analyze = data.get('crops', list(Config.CROP_SEASONS.keys())[:10])
        
        n = float(data.get('nitrogen', 80))
        p = float(data.get('phosphorus', 40))
        k = float(data.get('potassium', 40))
        ph = float(data.get('ph', 6.5))
        rainfall = float(data.get('rainfall', 100))
        temp = float(data.get('temperature', 28))
        soil_type = data.get('soil_type', 'loam')
        
        suitability_scores = []
        
        for crop in crops_to_analyze:
            crop_lower = crop.lower()
            
            # Soil health for this crop
            soil_score = soil_analyzer.calculate_soil_health_index(
                nitrogen=n, phosphorus=p, potassium=k, ph=ph,
                soil_type=soil_type
            )['soil_health_index']
            
            # Water requirement match
            water_need = Config.CROP_WATER_NEEDS.get(crop_lower, 500)
            water_score = min(100, (rainfall / water_need) * 100) if water_need > 0 else 50
            
            # Temperature suitability (simplified)
            temp_optimal = {
                'rice': (25, 35), 'wheat': (15, 25), 'maize': (20, 30),
                'cotton': (25, 35), 'sugarcane': (25, 35), 'potato': (15, 25),
                'tomato': (20, 30), 'onion': (15, 25), 'groundnut': (25, 35)
            }
            opt_range = temp_optimal.get(crop_lower, (20, 30))
            if opt_range[0] <= temp <= opt_range[1]:
                temp_score = 100
            else:
                distance = min(abs(temp - opt_range[0]), abs(temp - opt_range[1]))
                temp_score = max(30, 100 - distance * 5)
            
            # Season suitability
            current_month = datetime.now().month
            seasons = Config.CROP_SEASONS.get(crop_lower, {})
            season_score = 50
            for season_name, (start, end) in seasons.items():
                if start <= end:
                    if start <= current_month <= end:
                        season_score = 100
                        break
                else:
                    if current_month >= start or current_month <= end:
                        season_score = 100
                        break
            
            # Overall suitability
            overall = (soil_score * 0.35 + water_score * 0.25 + temp_score * 0.25 + season_score * 0.15)
            
            suitability_scores.append({
                'crop': crop,
                'overall_score': round(overall, 1),
                'soil_score': round(soil_score, 1),
                'water_score': round(water_score, 1),
                'temperature_score': round(temp_score, 1),
                'season_score': round(season_score, 1),
                'recommendation': 'Highly Suitable' if overall >= 75 else 'Suitable' if overall >= 55 else 'Marginal' if overall >= 40 else 'Not Recommended'
            })
        
        # Sort by overall score
        suitability_scores.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return jsonify({
            'success': True,
            'data': {
                'crops': suitability_scores,
                'best_crop': suitability_scores[0] if suitability_scores else None,
                'analysis_params': {
                    'nitrogen': n, 'phosphorus': p, 'potassium': k,
                    'ph': ph, 'rainfall': rainfall, 'temperature': temp,
                    'soil_type': soil_type
                }
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/ml/profit-calculator', methods=['POST'])
def calculate_profit():
    """
    Comprehensive profit/loss calculator for crop planning
    """
    data = request.json
    
    try:
        crop = data.get('crop', 'rice').lower()
        area = float(data.get('area_hectares', 1))
        
        # Get yield prediction
        yield_result = yield_predictor.predict_yield(
            crop=crop,
            temp_avg=float(data.get('temp_avg', 28)),
            rainfall=float(data.get('rainfall', 100)),
            soil_n=float(data.get('nitrogen', 80)),
            soil_p=float(data.get('phosphorus', 40)),
            soil_k=float(data.get('potassium', 40)),
            soil_ph=float(data.get('ph', 6.5))
        )
        
        predicted_yield = yield_result['predicted_yield']
        
        # Market price
        market_price = float(data.get('market_price', Config.FALLBACK_CROP_PRICES.get(crop, 2000)))
        
        # Calculate revenue
        total_yield_kg = predicted_yield * area
        total_yield_quintals = total_yield_kg / 100
        gross_revenue = total_yield_quintals * market_price
        
        # Costs estimation
        fertilizer_needs = Config.CROP_FERTILIZER.get(crop, {'N': 100, 'P': 50, 'K': 50})
        
        costs = {
            'seed': float(data.get('seed_cost', 3000 * area)),
            'fertilizer': float(data.get('fertilizer_cost', (
                (fertilizer_needs['N'] * area * 15) +  # Urea ~₹15/kg N
                (fertilizer_needs['P'] * area * 25) +  # DAP ~₹25/kg P
                (fertilizer_needs['K'] * area * 20)    # MOP ~₹20/kg K
            ))),
            'pesticides': float(data.get('pesticide_cost', 2500 * area)),
            'irrigation': float(data.get('irrigation_cost', 5000 * area)),
            'labor': float(data.get('labor_cost', 15000 * area)),
            'machinery': float(data.get('machinery_cost', 8000 * area)),
            'transport': float(data.get('transport_cost', 1000 * area)),
            'other': float(data.get('other_cost', 2000 * area))
        }
        
        total_cost = sum(costs.values())
        net_profit = gross_revenue - total_cost
        profit_margin = (net_profit / gross_revenue * 100) if gross_revenue > 0 else 0
        roi = (net_profit / total_cost * 100) if total_cost > 0 else 0
        
        return jsonify({
            'success': True,
            'data': {
                'crop': crop,
                'area_hectares': area,
                'yield': {
                    'predicted_kg_per_ha': predicted_yield,
                    'total_kg': total_yield_kg,
                    'total_quintals': total_yield_quintals,
                    'confidence': yield_result['confidence']
                },
                'revenue': {
                    'market_price_per_quintal': market_price,
                    'gross_revenue': round(gross_revenue, 0)
                },
                'costs': costs,
                'total_cost': round(total_cost, 0),
                'financials': {
                    'net_profit': round(net_profit, 0),
                    'profit_margin_percent': round(profit_margin, 1),
                    'return_on_investment_percent': round(roi, 1),
                    'break_even_yield_kg': round(total_cost / (market_price / 100), 0) if market_price > 0 else 0
                },
                'verdict': 'Profitable' if net_profit > 0 else 'Loss Expected'
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/ml/historical-weather', methods=['GET'])
def get_historical_weather_analysis():
    """
    Analyze historical weather patterns (simulated for demo)
    """
    lat = request.args.get('lat', 20)
    lon = request.args.get('lon', 77)
    
    # Generate simulated historical patterns for Indian agriculture
    # In production, this would use actual historical data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Typical patterns for central India
    monthly_data = {
        'temperature': {
            'avg': [18, 21, 27, 33, 38, 35, 30, 28, 29, 28, 23, 19],
            'max': [25, 29, 35, 40, 44, 40, 35, 32, 33, 33, 30, 26],
            'min': [10, 13, 18, 24, 28, 27, 25, 24, 24, 22, 16, 11]
        },
        'rainfall': [10, 8, 5, 3, 15, 150, 300, 280, 180, 50, 15, 8],
        'humidity': [55, 45, 35, 30, 40, 70, 85, 88, 82, 65, 55, 58],
        'gdd_accumulation': [180, 200, 350, 450, 550, 480, 400, 380, 400, 420, 320, 250]
    }
    
    # Season analysis
    seasons = {
        'kharif': {
            'months': 'June - October',
            'avg_rainfall': sum(monthly_data['rainfall'][5:10]),
            'suitable_crops': ['Rice', 'Maize', 'Cotton', 'Soybean', 'Groundnut']
        },
        'rabi': {
            'months': 'November - March',
            'avg_rainfall': sum(monthly_data['rainfall'][:3] + monthly_data['rainfall'][10:]),
            'suitable_crops': ['Wheat', 'Chickpea', 'Mustard', 'Barley']
        },
        'zaid': {
            'months': 'March - June',
            'avg_rainfall': sum(monthly_data['rainfall'][2:6]),
            'suitable_crops': ['Watermelon', 'Cucumber', 'Muskmelon']
        }
    }
    
    return jsonify({
        'success': True,
        'data': {
            'location': {'lat': lat, 'lon': lon},
            'months': months,
            'monthly_data': monthly_data,
            'seasons': seasons,
            'monsoon_analysis': {
                'onset': 'First week of June (typical)',
                'withdrawal': 'Mid-October (typical)',
                'total_seasonal_rainfall': sum(monthly_data['rainfall'][5:10]),
                'peak_month': 'July'
            },
            'farming_calendar_recommendations': {
                'kharif_sowing': 'June 15 - July 15',
                'rabi_sowing': 'October 25 - November 25',
                'wheat_harvest': 'March 15 - April 15',
                'rice_harvest': 'October - November'
            }
        }
    })


# ============================================================================
# AI ASSISTANT ENDPOINTS (Ollama Integration)
# ============================================================================

@app.route('/api/ai/status')
def ai_status():
    """Check AI assistant availability"""
    status = ai_assistant.check_connection()
    return jsonify(status)


@app.route('/api/ai/chat', methods=['POST'])
def ai_chat():
    """Chat with AI assistant"""
    data = request.json
    message = data.get('message', '')
    conversation_history = data.get('history', [])
    language = data.get('language', session.get('language', 'en'))
    
    # Get user context
    context = {}
    user_id = session.get('user_id')
    if user_id:
        user = User.query.get(user_id)
        if user:
            context = {
                'location': f"{user.city or user.district}, {user.state}",
                'soil_type': user.soil_type,
                'farm_size': user.farm_size
            }
    
    # Add weather context if available
    if data.get('include_weather') and context.get('location'):
        try:
            weather_url = f'/api/weather?city={context["location"]}'
            # We can't call our own API internally, so skip this for now
            pass
        except:
            pass
    
    result = ai_assistant.chat(
        message=message,
        conversation_history=conversation_history,
        language=language,
        context=context
    )
    
    return jsonify(result)


@app.route('/api/ai/chat/stream', methods=['POST'])
def ai_chat_stream():
    """Stream chat response from AI assistant"""
    data = request.json
    message = data.get('message', '')
    conversation_history = data.get('history', [])
    language = data.get('language', session.get('language', 'en'))
    
    context = {}
    user_id = session.get('user_id')
    if user_id:
        user = User.query.get(user_id)
        if user:
            context = {
                'location': f"{user.city or user.district}, {user.state}",
                'soil_type': user.soil_type,
                'farm_size': user.farm_size
            }
    
    def generate():
        for chunk in ai_assistant.chat_stream(
            message=message,
            conversation_history=conversation_history,
            language=language,
            context=context
        ):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/ai/suggestions')
def ai_suggestions():
    """Get quick suggestion prompts"""
    language = request.args.get('lang', session.get('language', 'en'))
    suggestions = ai_assistant.get_quick_suggestions(language)
    return jsonify({'success': True, 'suggestions': suggestions})


# ============================================================================
# DISEASE DETECTION ENDPOINTS
# ============================================================================

@app.route('/api/disease/detect', methods=['POST'])
def detect_disease():
    """Detect plant disease from image"""
    language = request.form.get('language', session.get('language', 'en'))
    
    # Handle image upload
    if 'image' in request.files:
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'}), 400
        
        image_data = image_file.read()
    elif 'image_base64' in request.form:
        # Handle base64 encoded image
        image_b64 = request.form['image_base64']
        # Remove data URL prefix if present
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        image_data = base64.b64decode(image_b64)
    elif request.is_json:
        data = request.json
        # Accept both 'image' and 'image_base64' keys
        image_b64 = data.get('image') or data.get('image_base64')
        if image_b64:
            if ',' in image_b64:
                image_b64 = image_b64.split(',')[1]
            image_data = base64.b64decode(image_b64)
            language = data.get('lang', language)
        else:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
    else:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    # Detect disease
    result = disease_detector.detect_disease(image_data, language)
    return jsonify(result)


@app.route('/api/disease/info/<disease_key>')
def get_disease_info(disease_key):
    """Get detailed information about a disease"""
    language = request.args.get('lang', session.get('language', 'en'))
    result = disease_detector.get_disease_info(disease_key, language)
    return jsonify(result)


@app.route('/api/disease/list')
def list_diseases():
    """Get list of all detectable diseases"""
    language = request.args.get('lang', session.get('language', 'en'))
    diseases = disease_detector.get_all_diseases(language)
    return jsonify({'success': True, 'diseases': diseases})


# ============================================================================
# PLOT ANALYSIS ENDPOINTS (For Map-based Recommendations)
# ============================================================================

@app.route('/api/plot/analyze', methods=['POST'])
def analyze_plot():
    """
    Analyze a farm plot based on boundary coordinates
    Returns zone-wise planting recommendations
    """
    data = request.json
    lang = data.get('lang', session.get('language', 'en'))
    
    # Get plot boundary coordinates (array of [lat, lon] pairs)
    coordinates = data.get('coordinates', [])
    if len(coordinates) < 3:
        return jsonify({'success': False, 'error': 'At least 3 coordinates required to define a plot'}), 400
    
    # Calculate plot area (simplified - using shoelace formula)
    n = len(coordinates)
    area_deg = 0
    for i in range(n):
        j = (i + 1) % n
        area_deg += coordinates[i][0] * coordinates[j][1]
        area_deg -= coordinates[j][0] * coordinates[i][1]
    area_deg = abs(area_deg) / 2
    
    # Convert to hectares (approximate at India's latitude)
    # 1 degree ≈ 111km at equator, varies with latitude
    avg_lat = sum(c[0] for c in coordinates) / n
    km_per_deg_lat = 111
    km_per_deg_lon = 111 * np.cos(np.radians(avg_lat))
    area_km2 = area_deg * km_per_deg_lat * km_per_deg_lon
    area_hectares = area_km2 * 100
    
    # Get centroid
    centroid_lat = sum(c[0] for c in coordinates) / n
    centroid_lon = sum(c[1] for c in coordinates) / n
    
    # Get user's soil type and other info
    soil_type = data.get('soil_type', 'loam')
    nitrogen = data.get('nitrogen', 80)
    phosphorus = data.get('phosphorus', 40)
    potassium = data.get('potassium', 40)
    ph = data.get('ph', 6.5)
    
    # Determine current season
    current_month = datetime.now().month
    if 6 <= current_month <= 10:
        current_season = 'kharif'
        season_crops = ['rice', 'maize', 'cotton', 'soybean', 'groundnut', 'sugarcane']
    elif current_month >= 11 or current_month <= 3:
        current_season = 'rabi'
        season_crops = ['wheat', 'chickpea', 'mustard', 'potato', 'onion', 'tomato']
    else:
        current_season = 'zaid'
        season_crops = ['watermelon', 'cucumber', 'muskmelon', 'vegetables']
    
    # Analyze suitability for different crops
    crop_recommendations = []
    for crop in season_crops:
        soil_health = soil_analyzer.calculate_soil_health_index(
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            potassium=potassium,
            ph=ph,
            soil_type=soil_type
        )
        
        water_need = Config.CROP_WATER_NEEDS.get(crop, 500)
        
        crop_recommendations.append({
            'crop': crop,
            'crop_translated': get_crop_name(crop, lang),
            'suitability_score': soil_health['soil_health_index'],
            'water_requirement_mm': water_need,
            'recommended_area_percent': min(40, max(10, soil_health['soil_health_index'] / 3)),
            'expected_yield_per_ha': Config.BASE_YIELDS.get(crop, 2000)
        })
    
    # Sort by suitability
    crop_recommendations.sort(key=lambda x: x['suitability_score'], reverse=True)
    
    # Create zone recommendations (divide plot into zones)
    zones = []
    top_crops = crop_recommendations[:3]
    total_percent = sum(c['recommended_area_percent'] for c in top_crops)
    
    for i, crop in enumerate(top_crops):
        normalized_percent = (crop['recommended_area_percent'] / total_percent) * 100 if total_percent > 0 else 33
        zones.append({
            'zone_id': i + 1,
            'zone_name': f"Zone {i + 1}",
            'recommended_crop': crop['crop'],
            'recommended_crop_translated': crop['crop_translated'],
            'area_percent': round(normalized_percent, 1),
            'area_hectares': round(area_hectares * normalized_percent / 100, 2),
            'expected_yield_kg': round(crop['expected_yield_per_ha'] * area_hectares * normalized_percent / 100, 0),
            'water_requirement': round(crop['water_requirement_mm'] * area_hectares * normalized_percent / 100 * 10, 0),  # liters
            'suitability': crop['suitability_score']
        })
    
    return jsonify({
        'success': True,
        'plot_info': {
            'area_hectares': round(area_hectares, 2),
            'centroid': {'lat': centroid_lat, 'lon': centroid_lon},
            'coordinates_count': len(coordinates),
            'current_season': current_season
        },
        'zones': zones,
        'all_recommendations': crop_recommendations,
        'soil_analysis': {
            'nitrogen': nitrogen,
            'phosphorus': phosphorus,
            'potassium': potassium,
            'ph': ph,
            'soil_type': soil_type
        }
    })


@app.route('/api/user/plots', methods=['GET'])
def get_user_plots():
    """Get all plots for the current user"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    plots = FarmPlot.query.filter_by(user_id=user_id).all()
    return jsonify({
        'success': True,
        'plots': [p.to_dict() for p in plots]
    })


@app.route('/api/user/plots', methods=['POST'])
def save_user_plot():
    """Save a new plot for the user"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    data = request.json
    coordinates = data.get('coordinates', [])
    
    if len(coordinates) < 3:
        return jsonify({'success': False, 'error': 'At least 3 coordinates required'}), 400
    
    # Calculate area and centroid
    n = len(coordinates)
    area_deg = 0
    for i in range(n):
        j = (i + 1) % n
        area_deg += coordinates[i][0] * coordinates[j][1]
        area_deg -= coordinates[j][0] * coordinates[i][1]
    area_deg = abs(area_deg) / 2
    
    avg_lat = sum(c[0] for c in coordinates) / n
    avg_lng = sum(c[1] for c in coordinates) / n
    km_per_deg_lat = 111
    km_per_deg_lon = 111 * np.cos(np.radians(avg_lat))
    area_km2 = area_deg * km_per_deg_lat * km_per_deg_lon
    area_hectares = area_km2 * 100
    
    import json
    plot = FarmPlot(
        user_id=user_id,
        name=data.get('name', f"Plot {FarmPlot.query.filter_by(user_id=user_id).count() + 1}"),
        coordinates=json.dumps(coordinates),
        area_hectares=round(area_hectares, 2),
        center_lat=avg_lat,
        center_lng=avg_lng,
        detected_soil_type=data.get('soil_type'),
        current_crop=data.get('current_crop'),
        analysis_source='user'
    )
    
    db.session.add(plot)
    db.session.commit()
    
    # Also update user's farm_size with total area
    user = User.query.get(user_id)
    total_area = db.session.query(db.func.sum(FarmPlot.area_hectares)).filter_by(user_id=user_id).scalar() or 0
    user.farm_size = total_area
    db.session.commit()
    
    return jsonify({
        'success': True,
        'plot': plot.to_dict()
    })


@app.route('/api/user/plots/<int:plot_id>', methods=['DELETE'])
def delete_user_plot(plot_id):
    """Delete a user's plot"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    plot = FarmPlot.query.filter_by(id=plot_id, user_id=user_id).first()
    if not plot:
        return jsonify({'success': False, 'error': 'Plot not found'}), 404
    
    db.session.delete(plot)
    db.session.commit()
    
    return jsonify({'success': True})


@app.route('/api/plot/satellite-info', methods=['POST'])
def get_satellite_info():
    """
    Get satellite/map tile information for a location
    Returns URLs for different map layers
    """
    data = request.json
    lat = data.get('lat')
    lon = data.get('lon')
    zoom = data.get('zoom', 15)
    
    if not lat or not lon:
        return jsonify({'success': False, 'error': 'Latitude and longitude required'}), 400
    
    # Calculate tile coordinates
    import math
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    
    return jsonify({
        'success': True,
        'tile_info': {
            'x': x,
            'y': y,
            'zoom': zoom
        },
        'map_layers': {
            'osm': f'https://tile.openstreetmap.org/{zoom}/{x}/{y}.png',
            'satellite': f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}',
            'terrain': f'https://stamen-tiles.a.ssl.fastly.net/terrain/{zoom}/{x}/{y}.png'
        }
    })


# ============================================================================
# VOICE ASSISTANT SUPPORT
# ============================================================================

@app.route('/api/voice/process', methods=['POST'])
def process_voice_command():
    """
    Process voice command and return appropriate action
    """
    data = request.json
    text = data.get('text', '').lower()
    language = data.get('language', 'en')
    
    # Define command patterns
    commands = {
        'weather': {
            'en': ['weather', 'temperature', 'rain', 'forecast'],
            'hi': ['मौसम', 'तापमान', 'बारिश'],
            'kn': ['ಹವಾಮಾನ', 'ತಾಪಮಾನ', 'ಮಳೆ']
        },
        'crop': {
            'en': ['crop', 'plant', 'recommend', 'grow', 'sow'],
            'hi': ['फसल', 'बुवाई', 'उगाना'],
            'kn': ['ಬೆಳೆ', 'ಬಿತ್ತನೆ']
        },
        'price': {
            'en': ['price', 'market', 'mandi', 'sell'],
            'hi': ['भाव', 'बाजार', 'मंडी'],
            'kn': ['ಬೆಲೆ', 'ಮಾರುಕಟ್ಟೆ']
        },
        'disease': {
            'en': ['disease', 'pest', 'infection', 'sick'],
            'hi': ['रोग', 'कीट', 'बीमारी'],
            'kn': ['ರೋಗ', 'ಕೀಟ']
        },
        'water': {
            'en': ['water', 'irrigation', 'irrigate'],
            'hi': ['पानी', 'सिंचाई'],
            'kn': ['ನೀರು', 'ನೀರಾವರಿ']
        },
        'scheme': {
            'en': ['scheme', 'government', 'subsidy', 'loan'],
            'hi': ['योजना', 'सरकारी', 'सब्सिडी'],
            'kn': ['ಯೋಜನೆ', 'ಸರ್ಕಾರಿ']
        }
    }
    
    # Detect command type
    detected_command = None
    for cmd_type, patterns in commands.items():
        lang_patterns = patterns.get(language, patterns.get('en', []))
        if any(pattern in text for pattern in lang_patterns):
            detected_command = cmd_type
            break
    
    # Generate response and action
    response = {
        'success': True,
        'detected_command': detected_command,
        'original_text': text,
        'language': language
    }
    
    if detected_command:
        actions = {
            'weather': {'action': 'show_weather', 'speak': get_translation('current_weather', language)},
            'crop': {'action': 'show_crop_recommendation', 'speak': get_translation('crop_recommendation', language)},
            'price': {'action': 'show_market_prices', 'speak': get_translation('market_prices', language)},
            'disease': {'action': 'show_disease_scanner', 'speak': 'Opening disease scanner'},
            'water': {'action': 'show_water_calculator', 'speak': get_translation('water_calculator', language)},
            'scheme': {'action': 'show_schemes', 'speak': get_translation('gov_schemes', language)}
        }
        response.update(actions.get(detected_command, {}))
    else:
        # If no specific command, use AI assistant
        response['action'] = 'ai_chat'
        response['ai_query'] = text
    
    return jsonify(response)


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Endpoint not found'}), 404
    return redirect(url_for('setup'))


@app.errorhandler(500)
def server_error(e):
    lang = session.get('language', 'en')
    if request.path.startswith('/api/'):
        return jsonify({
            'error': True,
            'message': 'Server error. Please try again.'
        }), 500
    return render_template('error.html', message=get_translation('error_weather', lang)), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
