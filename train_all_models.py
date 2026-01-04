"""
CERES Advanced ML Model Training Pipeline
==========================================
Trains multiple ML models for agricultural predictions:
1. Crop Recommendation Model (RandomForest)
2. Yield Prediction Model (GradientBoosting)
3. Disease Risk Model (XGBoost-style)
4. Soil Health Score Model (Neural Network-style)

Uses both synthetic and real datasets.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, mean_squared_error, 
    r2_score, mean_absolute_error, confusion_matrix
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def generate_crop_recommendation_data(n_samples=3000):
    """Generate synthetic data for crop recommendation model"""
    print("📊 Generating crop recommendation dataset...")
    
    np.random.seed(42)
    
    # Crop parameters: [N_range, P_range, K_range, temp_range, humidity_range, ph_range, rainfall_range]
    crop_params = {
        'rice': ([60, 100], [35, 65], [35, 55], [20, 35], [70, 95], [5.5, 7.0], [150, 300]),
        'wheat': ([80, 140], [40, 70], [30, 60], [10, 26], [50, 75], [5.5, 7.5], [50, 120]),
        'maize': ([60, 120], [35, 65], [25, 55], [18, 32], [55, 80], [5.5, 7.5], [60, 120]),
        'cotton': ([100, 160], [45, 75], [45, 75], [25, 40], [50, 70], [5.8, 8.0], [50, 120]),
        'sugarcane': ([80, 150], [40, 80], [40, 80], [20, 38], [70, 90], [5.0, 8.0], [100, 220]),
        'potato': ([120, 200], [60, 100], [80, 140], [10, 25], [60, 85], [4.5, 6.5], [50, 120]),
        'tomato': ([80, 150], [50, 100], [60, 120], [18, 32], [55, 80], [5.5, 7.0], [40, 100]),
        'groundnut': ([15, 40], [40, 70], [30, 60], [24, 35], [55, 75], [5.5, 7.0], [50, 100]),
        'soybean': ([20, 50], [45, 80], [20, 50], [18, 30], [55, 75], [5.6, 7.0], [60, 120]),
        'chickpea': ([15, 35], [40, 75], [20, 50], [15, 28], [40, 65], [5.5, 7.5], [30, 80]),
        'mustard': ([50, 90], [30, 55], [20, 45], [12, 25], [40, 65], [5.5, 7.5], [25, 60]),
        'lentil': ([10, 30], [35, 60], [15, 40], [12, 25], [40, 60], [5.5, 7.5], [25, 60]),
        'mango': ([40, 80], [30, 60], [50, 100], [22, 38], [50, 80], [5.5, 7.5], [100, 200]),
        'banana': ([100, 180], [30, 70], [180, 300], [22, 35], [70, 95], [5.5, 7.0], [120, 250]),
        'onion': ([60, 110], [35, 65], [35, 65], [15, 30], [55, 75], [5.8, 7.0], [40, 90]),
        'jute': ([50, 90], [25, 50], [35, 65], [25, 38], [70, 95], [5.5, 7.0], [150, 300]),
        'coffee': ([60, 120], [25, 55], [80, 140], [18, 28], [65, 90], [5.0, 6.5], [120, 200]),
        'millets': ([30, 70], [20, 45], [20, 45], [22, 38], [45, 70], [5.5, 7.5], [30, 80]),
        'turmeric': ([80, 140], [40, 80], [80, 140], [22, 32], [70, 90], [5.0, 7.0], [100, 180]),
        'cardamom': ([80, 140], [35, 65], [100, 180], [16, 26], [75, 95], [5.0, 6.5], [150, 300]),
        'pepper': ([70, 130], [30, 60], [80, 150], [22, 32], [70, 95], [5.0, 6.5], [200, 350]),
        'coconut': ([50, 100], [20, 50], [80, 160], [24, 36], [65, 95], [5.5, 8.0], [100, 250]),
    }
    
    data = []
    for crop, params in crop_params.items():
        n_per_crop = n_samples // len(crop_params)
        
        for _ in range(n_per_crop):
            # Add some noise and variability
            noise_factor = np.random.uniform(0.8, 1.2)
            
            n = np.random.uniform(*params[0]) * noise_factor
            p = np.random.uniform(*params[1]) * noise_factor
            k = np.random.uniform(*params[2]) * noise_factor
            temp = np.random.uniform(*params[3])
            humidity = np.random.uniform(*params[4])
            ph = np.random.uniform(*params[5])
            rainfall = np.random.uniform(*params[6]) * noise_factor
            
            data.append([n, p, k, temp, humidity, ph, rainfall, crop])
    
    df = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'crop'])
    return df


def train_crop_recommendation_model():
    """Train the crop recommendation model"""
    print("\n" + "="*60)
    print("🌾 TRAINING CROP RECOMMENDATION MODEL")
    print("="*60)
    
    df = generate_crop_recommendation_data(3500)
    
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['crop']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"📈 Training samples: {len(X_train)}")
    print(f"📊 Test samples: {len(X_test)}")
    print(f"🌿 Number of crops: {len(label_encoder.classes_)}")
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [15, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    print("\n🔧 Performing hyperparameter tuning...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"✅ Best parameters: {grid_search.best_params_}")
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"   Accuracy: {accuracy * 100:.2f}%")
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, X_scaled, y_encoded, cv=5)
    print(f"   Cross-validation: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    
    # Feature importance
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    importance = best_model.feature_importances_
    print(f"\n📊 Feature Importance:")
    for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
        print(f"   {name}: {imp*100:.1f}%")
    
    # Save models
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'crop_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    
    print(f"\n✅ Models saved to {MODELS_DIR}")
    return accuracy


def train_yield_prediction_model():
    """Train yield prediction model using GradientBoosting"""
    print("\n" + "="*60)
    print("📈 TRAINING YIELD PREDICTION MODEL")
    print("="*60)
    
    # Try to load real dataset
    dataset_path = os.path.join(DATA_DIR, 'crop_yield_dataset.csv')
    
    if os.path.exists(dataset_path):
        print(f"📁 Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        print("📊 Generating synthetic yield dataset...")
        df = generate_yield_data(1500)
    
    print(f"📈 Dataset size: {len(df)} records")
    print(f"🌾 Crops: {df['crop'].nunique()}")
    
    # Feature engineering
    le_crop = LabelEncoder()
    le_soil = LabelEncoder()
    le_irrigation = LabelEncoder()
    
    df['crop_encoded'] = le_crop.fit_transform(df['crop'])
    df['soil_encoded'] = le_soil.fit_transform(df['soil_type'])
    df['irrigation_encoded'] = le_irrigation.fit_transform(df['irrigation_type'])
    df['pesticide_encoded'] = df['pesticide_applied'].map({'yes': 1, 'no': 0})
    
    features = ['crop_encoded', 'temperature_avg', 'rainfall_mm', 'humidity_avg', 
                'soil_encoded', 'nitrogen_kg', 'phosphorus_kg', 'potassium_kg', 
                'ph', 'organic_carbon', 'irrigation_encoded', 'pesticide_encoded']
    
    X = df[features]
    y = df['yield_kg_per_ha']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train Gradient Boosting model
    print("\n🔧 Training GradientBoostingRegressor...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: {rmse:.2f} kg/ha")
    print(f"   MAE: {mae:.2f} kg/ha")
    print(f"   Mean Yield: {y.mean():.2f} kg/ha")
    print(f"   Error %: {(mae/y.mean())*100:.1f}%")
    
    # Feature importance
    print(f"\n📊 Feature Importance:")
    for name, imp in sorted(zip(features, model.feature_importances_), 
                            key=lambda x: x[1], reverse=True)[:8]:
        print(f"   {name}: {imp*100:.1f}%")
    
    # Save models
    joblib.dump(model, os.path.join(MODELS_DIR, 'yield_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'yield_scaler.pkl'))
    joblib.dump({
        'crop': le_crop,
        'soil': le_soil,
        'irrigation': le_irrigation
    }, os.path.join(MODELS_DIR, 'yield_encoders.pkl'))
    
    print(f"\n✅ Yield model saved to {MODELS_DIR}")
    return r2


def generate_yield_data(n_samples):
    """Generate synthetic yield data"""
    np.random.seed(42)
    
    crops = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'potato', 
             'tomato', 'onion', 'groundnut', 'soybean']
    soil_types = ['alluvial', 'black', 'red', 'laterite', 'arid']
    irrigation_types = ['canal', 'tube_well', 'flood', 'drip', 'rainfed']
    
    base_yields = {
        'rice': 5500, 'wheat': 4500, 'maize': 7000, 'cotton': 2000,
        'sugarcane': 75000, 'potato': 25000, 'tomato': 40000,
        'onion': 18000, 'groundnut': 2500, 'soybean': 2800
    }
    
    data = []
    for _ in range(n_samples):
        crop = np.random.choice(crops)
        base = base_yields[crop]
        
        temp = np.random.uniform(15, 35)
        rainfall = np.random.uniform(200, 1200)
        humidity = np.random.uniform(40, 90)
        soil = np.random.choice(soil_types)
        n = np.random.uniform(50, 200)
        p = np.random.uniform(20, 100)
        k = np.random.uniform(30, 150)
        ph = np.random.uniform(5.5, 8.0)
        oc = np.random.uniform(0.3, 1.2)
        irrigation = np.random.choice(irrigation_types)
        pesticide = np.random.choice(['yes', 'no'])
        
        # Calculate yield with factors
        yield_factor = 1.0
        
        # Temperature effect
        if 20 <= temp <= 30:
            yield_factor *= 1.0
        else:
            yield_factor *= 0.85
        
        # Soil effect
        soil_factors = {'alluvial': 1.1, 'black': 1.0, 'red': 0.9, 'laterite': 0.85, 'arid': 0.7}
        yield_factor *= soil_factors.get(soil, 0.9)
        
        # Irrigation effect
        irr_factors = {'drip': 1.15, 'canal': 1.05, 'tube_well': 1.0, 'flood': 0.95, 'rainfed': 0.8}
        yield_factor *= irr_factors.get(irrigation, 0.9)
        
        # Nutrient effect
        if n >= 100 and p >= 50 and k >= 60:
            yield_factor *= 1.05
        
        # Pesticide effect
        if pesticide == 'yes':
            yield_factor *= 1.08
        
        # Add noise
        yield_factor *= np.random.uniform(0.85, 1.15)
        
        final_yield = base * yield_factor
        
        data.append({
            'crop': crop,
            'temperature_avg': temp,
            'rainfall_mm': rainfall,
            'humidity_avg': humidity,
            'soil_type': soil,
            'nitrogen_kg': n,
            'phosphorus_kg': p,
            'potassium_kg': k,
            'ph': ph,
            'organic_carbon': oc,
            'irrigation_type': irrigation,
            'pesticide_applied': pesticide,
            'yield_kg_per_ha': final_yield
        })
    
    return pd.DataFrame(data)


def train_disease_risk_model():
    """Train disease risk prediction model"""
    print("\n" + "="*60)
    print("🦠 TRAINING DISEASE RISK MODEL")
    print("="*60)
    
    # Try to load real dataset
    dataset_path = os.path.join(DATA_DIR, 'disease_risk_dataset.csv')
    
    if os.path.exists(dataset_path):
        print(f"📁 Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        print("📊 Generating synthetic disease risk dataset...")
        df = generate_disease_data(1000)
    
    print(f"📈 Dataset size: {len(df)} records")
    print(f"🦠 Diseases: {df['disease'].nunique()}")
    
    # Feature engineering
    le_crop = LabelEncoder()
    le_disease = LabelEncoder()
    
    df['crop_encoded'] = le_crop.fit_transform(df['crop'])
    df['disease_encoded'] = le_disease.fit_transform(df['disease'])
    
    features = ['crop_encoded', 'temperature', 'humidity', 'rainfall_mm', 
                'wind_speed', 'dew_hours', 'consecutive_wet_days', 'soil_moisture']
    
    X = df[features]
    y = df['severity_score']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("\n🔧 Training RandomForestRegressor for disease risk...")
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAE: {mae:.2f}")
    
    # Save models
    joblib.dump(model, os.path.join(MODELS_DIR, 'disease_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'disease_scaler.pkl'))
    joblib.dump({
        'crop': le_crop,
        'disease': le_disease
    }, os.path.join(MODELS_DIR, 'disease_encoders.pkl'))
    
    print(f"\n✅ Disease risk model saved to {MODELS_DIR}")
    return r2


def generate_disease_data(n_samples):
    """Generate synthetic disease risk data"""
    np.random.seed(42)
    
    crops = ['rice', 'wheat', 'potato', 'tomato', 'cotton', 'maize', 
             'sugarcane', 'groundnut', 'soybean', 'onion']
    
    crop_diseases = {
        'rice': ['bacterial_leaf_blight', 'brown_spot', 'sheath_blight', 'blast'],
        'wheat': ['rust', 'powdery_mildew', 'karnal_bunt'],
        'potato': ['late_blight', 'early_blight'],
        'tomato': ['late_blight', 'early_blight', 'powdery_mildew', 'bacterial_wilt'],
        'cotton': ['bacterial_blight', 'alternaria_leaf_spot', 'fusarium_wilt'],
        'maize': ['northern_leaf_blight', 'southern_leaf_blight', 'rust'],
        'sugarcane': ['red_rot', 'smut', 'wilt'],
        'groundnut': ['tikka_disease', 'rust', 'collar_rot'],
        'soybean': ['rust', 'anthracnose'],
        'onion': ['purple_blotch', 'downy_mildew', 'basal_rot']
    }
    
    data = []
    for _ in range(n_samples):
        crop = np.random.choice(crops)
        disease = np.random.choice(crop_diseases[crop])
        
        temp = np.random.uniform(15, 35)
        humidity = np.random.uniform(40, 95)
        rainfall = np.random.uniform(0, 30)
        wind = np.random.uniform(1, 8)
        dew = np.random.uniform(0, 10)
        wet_days = np.random.randint(0, 8)
        soil_moisture = np.random.uniform(30, 90)
        
        # Calculate severity based on conditions
        severity = 20  # Base
        
        # Humidity effect
        if humidity > 75:
            severity += (humidity - 75) * 0.8
        
        # Temperature effect (disease specific)
        if 18 <= temp <= 28:
            severity += 15
        
        # Rain effect
        if rainfall > 10:
            severity += (rainfall - 10) * 1.5
        
        # Wet days effect
        severity += wet_days * 5
        
        # Dew hours effect
        severity += dew * 2
        
        # Add noise
        severity += np.random.uniform(-10, 15)
        severity = max(10, min(100, severity))
        
        data.append({
            'crop': crop,
            'disease': disease,
            'temperature': temp,
            'humidity': humidity,
            'rainfall_mm': rainfall,
            'wind_speed': wind,
            'dew_hours': dew,
            'consecutive_wet_days': wet_days,
            'soil_moisture': soil_moisture,
            'severity_score': severity
        })
    
    return pd.DataFrame(data)


def train_soil_health_model():
    """Train soil health index prediction model"""
    print("\n" + "="*60)
    print("🌱 TRAINING SOIL HEALTH MODEL")
    print("="*60)
    
    # Try to load real dataset
    dataset_path = os.path.join(DATA_DIR, 'soil_health_dataset.csv')
    
    if os.path.exists(dataset_path):
        print(f"📁 Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        print("📊 Generating synthetic soil health dataset...")
        df = generate_soil_health_data(800)
    
    print(f"📈 Dataset size: {len(df)} records")
    
    # Feature engineering
    le_soil = LabelEncoder()
    le_texture = LabelEncoder()
    
    df['soil_encoded'] = le_soil.fit_transform(df['soil_type'])
    df['texture_encoded'] = le_texture.fit_transform(df['texture'])
    
    features = ['soil_encoded', 'ph', 'organic_carbon', 'nitrogen_kg_ha', 
                'phosphorus_kg_ha', 'potassium_kg_ha', 'ec_ds_m', 'texture_encoded']
    
    X = df[features]
    y = df['soil_health_index']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("\n🔧 Training GradientBoostingRegressor for soil health...")
    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAE: {mae:.2f}")
    
    # Feature importance
    print(f"\n📊 Feature Importance:")
    for name, imp in sorted(zip(features, model.feature_importances_), 
                            key=lambda x: x[1], reverse=True):
        print(f"   {name}: {imp*100:.1f}%")
    
    # Save models
    joblib.dump(model, os.path.join(MODELS_DIR, 'soil_health_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'soil_health_scaler.pkl'))
    joblib.dump({
        'soil': le_soil,
        'texture': le_texture
    }, os.path.join(MODELS_DIR, 'soil_health_encoders.pkl'))
    
    print(f"\n✅ Soil health model saved to {MODELS_DIR}")
    return r2


def generate_soil_health_data(n_samples):
    """Generate synthetic soil health data"""
    np.random.seed(42)
    
    soil_types = ['alluvial', 'black', 'red', 'laterite', 'arid', 'forest']
    textures = ['sandy', 'sandy_loam', 'loam', 'silty_loam', 'clay_loam', 'clay']
    
    soil_health_factors = {
        'alluvial': 1.1, 'black': 1.0, 'red': 0.9, 
        'laterite': 0.85, 'arid': 0.7, 'forest': 1.15
    }
    
    data = []
    for _ in range(n_samples):
        soil = np.random.choice(soil_types)
        texture = np.random.choice(textures)
        
        ph = np.random.uniform(5.0, 8.5)
        oc = np.random.uniform(0.2, 1.5)
        n = np.random.uniform(200, 550)
        p = np.random.uniform(15, 65)
        k = np.random.uniform(100, 250)
        ec = np.random.uniform(0.2, 1.0)
        
        # Calculate health index
        health = 50  # Base
        
        # pH effect (optimal 6.0-7.0)
        if 6.0 <= ph <= 7.0:
            health += 15
        elif 5.5 <= ph < 6.0 or 7.0 < ph <= 7.5:
            health += 8
        
        # Organic carbon effect
        health += min(15, oc * 12)
        
        # Nutrient effects
        if n >= 350:
            health += 8
        if p >= 35:
            health += 6
        if k >= 150:
            health += 5
        
        # Soil type effect
        health *= soil_health_factors.get(soil, 1.0)
        
        # EC penalty (high EC is bad)
        if ec > 0.7:
            health -= (ec - 0.7) * 20
        
        # Add noise
        health += np.random.uniform(-8, 8)
        health = max(25, min(95, health))
        
        data.append({
            'soil_type': soil,
            'texture': texture,
            'ph': ph,
            'organic_carbon': oc,
            'nitrogen_kg_ha': n,
            'phosphorus_kg_ha': p,
            'potassium_kg_ha': k,
            'ec_ds_m': ec,
            'soil_health_index': health
        })
    
    return pd.DataFrame(data)


def main():
    """Main training pipeline"""
    print("\n" + "🌾" * 30)
    print("  CERES-DTL ADVANCED ML TRAINING PIPELINE")
    print("🌾" * 30)
    
    results = {}
    
    # 1. Crop Recommendation Model
    results['crop_recommendation'] = train_crop_recommendation_model()
    
    # 2. Yield Prediction Model
    results['yield_prediction'] = train_yield_prediction_model()
    
    # 3. Disease Risk Model
    results['disease_risk'] = train_disease_risk_model()
    
    # 4. Soil Health Model
    results['soil_health'] = train_soil_health_model()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    print(f"✅ Crop Recommendation Accuracy: {results['crop_recommendation']*100:.2f}%")
    print(f"✅ Yield Prediction R²: {results['yield_prediction']:.4f}")
    print(f"✅ Disease Risk Prediction R²: {results['disease_risk']:.4f}")
    print(f"✅ Soil Health Prediction R²: {results['soil_health']:.4f}")
    print("\n🎉 All models trained and saved successfully!")
    print(f"📁 Models location: {MODELS_DIR}")


if __name__ == '__main__':
    main()
