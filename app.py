import os
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import numpy as np
import joblib

from config import Config
from database import db, init_db, User, SoilData, CropRecommendation, MarketPrice, FarmDiary, WeatherAlert, CropPlanning
from translations import TRANSLATIONS, CROP_TRANSLATIONS, LANGUAGE_NAMES, get_translation, get_crop_name

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
        return redirect(url_for('setup'))
    
    user = User.query.get(user_id)
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
    
    try:
        if lat and lon:
            pass
        elif city:
            geo_url = f"https://nominatim.openstreetmap.org/search?q={city},India&format=json&limit=1"
            headers = {'User-Agent': 'CERES-AgriTech/1.0'}
            geo_response = requests.get(geo_url, headers=headers, timeout=10)
            if geo_response.status_code == 200 and geo_response.json():
                geo_data = geo_response.json()[0]
                lat = geo_data['lat']
                lon = geo_data['lon']
            else:
                return jsonify({'error': True, 'message': get_translation('error_weather', lang)}), 400
        else:
            return jsonify({'error': True, 'message': get_translation('error_weather', lang)}), 400
        
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
        
        weather_icons = {
            0: '01d', 1: '02d', 2: '03d', 3: '04d', 45: '50d', 48: '50d',
            51: '09d', 53: '09d', 55: '09d', 61: '10d', 63: '10d', 65: '10d',
            71: '13d', 73: '13d', 75: '13d', 80: '09d', 81: '09d', 82: '09d', 95: '11d'
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
                'icon': weather_icons.get(weather_code, '01d'),
                'city': city or f"{lat}, {lon}"
            },
            'daily': {
                'sunrise': daily.get('sunrise', [None])[0],
                'sunset': daily.get('sunset', [None])[0],
            },
            'forecast': [],
            'timestamp': datetime.utcnow().isoformat()
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
                    'description': weather_descriptions.get(day_code, 'Unknown'),
                    'icon': weather_icons.get(day_code, '01d')
                })
        
        return jsonify(result)
        
    except requests.exceptions.RequestException as e:
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
                'message': 'High pest risk due to humid conditions',
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
        confidence = float(np.max(probabilities) * 100)
        
        top_3_indices = np.argsort(probabilities[0])[-3:][::-1]
        top_recommendations = [
            {
                'crop': LABEL_ENCODER.inverse_transform([idx])[0],
                'crop_translated': get_crop_name(LABEL_ENCODER.inverse_transform([idx])[0], lang),
                'confidence': float(probabilities[0][idx] * 100)
            }
            for idx in top_3_indices
        ]
        
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
            'confidence': round(confidence, 2),
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
    app.run(debug=True, host='0.0.0.0', port=5000)
