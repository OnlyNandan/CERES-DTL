# CERES - Smart Farming Platform for Indian Farmers

An intelligent Agri-Tech platform designed with "Simplicity for the user, Complexity in the backend."

## 🌾 Features

- **Multi-language Support**: Hindi, Kannada, Telugu, Tamil, Marathi, English
- **Real-time Weather**: OpenWeatherMap API integration
- **Market Prices**: Live APMC/Mandi prices from data.gov.in
- **ML Crop Recommendation**: RandomForest-based crop prediction
- **Geolocation**: Auto-detect user location
- **High-contrast UI**: Accessible design for all users

## 📁 Project Structure

```
CERES-DTL/
├── app.py                 # Flask backend with API routes
├── config.py              # Configuration settings
├── database.py            # SQLAlchemy models
├── translations.py        # Multi-language translations
├── model_train.py         # ML model training script
├── requirements.txt       # Python dependencies
├── data/
│   └── crop_data.csv      # Training dataset
├── models/                # Trained ML artifacts
│   ├── crop_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── templates/
│   ├── index.html         # Main dashboard
│   └── setup.html         # Onboarding page
└── static/
    └── js/
        └── main.js        # Frontend logic
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd CERES-DTL
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your API keys:
# - OPENWEATHERMAP_API_KEY (from openweathermap.org)
# - DATA_GOV_API_KEY (from data.gov.in)
```

### 3. Train the ML Model

```bash
python model_train.py
```

This generates:
- `models/crop_model.pkl` - Trained RandomForest model
- `models/scaler.pkl` - Feature scaler
- `models/label_encoder.pkl` - Label encoder

### 4. Run the Application

```bash
python app.py
```

Visit `http://localhost:5000`

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/weather` | GET | Fetch weather data |
| `/api/market-prices` | GET | Get APMC market prices |
| `/api/crop-recommendation` | POST | Get ML-based crop recommendation |
| `/api/translations` | GET | Get all translations |
| `/api/user/setup` | POST | Save user preferences |
| `/api/reverse-geocode` | GET | Convert coordinates to address |

## 🌐 Supported Languages

| Code | Language | Native |
|------|----------|--------|
| en | English | English |
| hi | Hindi | हिंदी |
| kn | Kannada | ಕನ್ನಡ |
| te | Telugu | తెలుగు |
| ta | Tamil | தமிழ் |
| mr | Marathi | मराठी |

## 🤖 ML Model Details

- **Algorithm**: RandomForest Classifier
- **Features**: N, P, K, Temperature, Humidity, pH, Rainfall
- **Crops**: 23 Indian crops (Rice, Wheat, Cotton, etc.)
- **Training Data**: 230 samples across soil conditions

### Crop Prediction API

```json
POST /api/crop-recommendation
{
    "N": 90,
    "P": 42,
    "K": 43,
    "ph": 6.5,
    "rainfall": 200,
    "temperature": 25,
    "humidity": 80
}
```

Response:
```json
{
    "success": true,
    "recommended_crop": "rice",
    "recommended_crop_translated": "चावल",
    "confidence": 92.5,
    "top_recommendations": [...]
}
```

## 📊 Data Sources

- **Weather**: [OpenWeatherMap API](https://openweathermap.org/api)
- **Market Prices**: [data.gov.in - APMC](https://data.gov.in/)
- **Geolocation**: [Nominatim/OpenStreetMap](https://nominatim.org/)

## 🛠️ Tech Stack

- **Backend**: Python, Flask, SQLAlchemy
- **Frontend**: HTML5, TailwindCSS, Vanilla JS
- **ML**: scikit-learn, pandas, numpy
- **Database**: SQLite (dev) / PostgreSQL (prod)

## 📝 License

MIT License - See LICENSE file
