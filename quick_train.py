#!/usr/bin/env python3
"""Quick ML Model Training Script"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import joblib

MODELS_DIR = 'models'
DATA_DIR = 'data'

os.makedirs(MODELS_DIR, exist_ok=True)

print('🌾 CERES ML Training (Fast Mode)')
print('='*50)

# 1. Train Crop Recommendation Model
print('\n📊 Training Crop Recommendation Model...')
np.random.seed(42)
crops = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'potato', 'tomato', 
         'groundnut', 'soybean', 'chickpea', 'mustard', 'onion', 'banana', 'mango']
data = []
for crop in crops:
    for _ in range(200):
        data.append([
            np.random.uniform(40, 160),
            np.random.uniform(25, 80),
            np.random.uniform(25, 100),
            np.random.uniform(15, 38),
            np.random.uniform(45, 95),
            np.random.uniform(5, 8),
            np.random.uniform(50, 250),
            crop
        ])

df = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'crop'])
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['crop']

le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
print(f'   Crop Model Accuracy: {acc*100:.2f}%')

joblib.dump(model, os.path.join(MODELS_DIR, 'crop_model.pkl'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.pkl'))

# 2. Train Yield Model
print('\n📈 Training Yield Prediction Model...')
yield_df = pd.read_csv(os.path.join(DATA_DIR, 'crop_yield_dataset.csv'))
le_crop = LabelEncoder()
le_soil = LabelEncoder()
le_irr = LabelEncoder()
yield_df['crop_enc'] = le_crop.fit_transform(yield_df['crop'])
yield_df['soil_enc'] = le_soil.fit_transform(yield_df['soil_type'])
yield_df['irr_enc'] = le_irr.fit_transform(yield_df['irrigation_type'])
yield_df['pest_enc'] = yield_df['pesticide_applied'].map({'yes': 1, 'no': 0})

X_yield = yield_df[['crop_enc', 'temperature_avg', 'rainfall_mm', 'humidity_avg', 'soil_enc', 
                    'nitrogen_kg', 'phosphorus_kg', 'potassium_kg', 'ph', 'organic_carbon', 'irr_enc', 'pest_enc']]
y_yield = yield_df['yield_kg_per_ha']

scaler_yield = StandardScaler()
X_yield_scaled = scaler_yield.fit_transform(X_yield)

X_tr, X_te, y_tr, y_te = train_test_split(X_yield_scaled, y_yield, test_size=0.2, random_state=42)
yield_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
yield_model.fit(X_tr, y_tr)
r2 = r2_score(y_te, yield_model.predict(X_te))
print(f'   Yield Model R2: {r2:.4f}')

joblib.dump(yield_model, os.path.join(MODELS_DIR, 'yield_model.pkl'))
joblib.dump(scaler_yield, os.path.join(MODELS_DIR, 'yield_scaler.pkl'))
joblib.dump({'crop': le_crop, 'soil': le_soil, 'irrigation': le_irr}, os.path.join(MODELS_DIR, 'yield_encoders.pkl'))

# 3. Train Disease Risk Model
print('\n🦠 Training Disease Risk Model...')
disease_df = pd.read_csv(os.path.join(DATA_DIR, 'disease_risk_dataset.csv'))
le_dcrop = LabelEncoder()
le_disease = LabelEncoder()
disease_df['crop_enc'] = le_dcrop.fit_transform(disease_df['crop'])
disease_df['disease_enc'] = le_disease.fit_transform(disease_df['disease'])

X_disease = disease_df[['crop_enc', 'temperature', 'humidity', 'rainfall_mm', 'wind_speed', 'dew_hours', 'consecutive_wet_days', 'soil_moisture']]
y_disease = disease_df['severity_score']

scaler_disease = StandardScaler()
X_disease_scaled = scaler_disease.fit_transform(X_disease)

X_tr, X_te, y_tr, y_te = train_test_split(X_disease_scaled, y_disease, test_size=0.2, random_state=42)
disease_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
disease_model.fit(X_tr, y_tr)
r2_d = r2_score(y_te, disease_model.predict(X_te))
print(f'   Disease Model R2: {r2_d:.4f}')

joblib.dump(disease_model, os.path.join(MODELS_DIR, 'disease_model.pkl'))
joblib.dump(scaler_disease, os.path.join(MODELS_DIR, 'disease_scaler.pkl'))
joblib.dump({'crop': le_dcrop, 'disease': le_disease}, os.path.join(MODELS_DIR, 'disease_encoders.pkl'))

# 4. Train Soil Health Model
print('\n🌱 Training Soil Health Model...')
soil_df = pd.read_csv(os.path.join(DATA_DIR, 'soil_health_dataset.csv'))
le_soiltype = LabelEncoder()
le_texture = LabelEncoder()
soil_df['soil_enc'] = le_soiltype.fit_transform(soil_df['soil_type'])
soil_df['texture_enc'] = le_texture.fit_transform(soil_df['texture'])

X_soil = soil_df[['soil_enc', 'ph', 'organic_carbon', 'nitrogen_kg_ha', 'phosphorus_kg_ha', 'potassium_kg_ha', 'ec_ds_m', 'texture_enc']]
y_soil = soil_df['soil_health_index']

scaler_soil = StandardScaler()
X_soil_scaled = scaler_soil.fit_transform(X_soil)

X_tr, X_te, y_tr, y_te = train_test_split(X_soil_scaled, y_soil, test_size=0.2, random_state=42)
soil_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
soil_model.fit(X_tr, y_tr)
r2_s = r2_score(y_te, soil_model.predict(X_te))
print(f'   Soil Health Model R2: {r2_s:.4f}')

joblib.dump(soil_model, os.path.join(MODELS_DIR, 'soil_health_model.pkl'))
joblib.dump(scaler_soil, os.path.join(MODELS_DIR, 'soil_health_scaler.pkl'))
joblib.dump({'soil': le_soiltype, 'texture': le_texture}, os.path.join(MODELS_DIR, 'soil_health_encoders.pkl'))

print('\n' + '='*50)
print('🎉 All models trained successfully!')
print(f'📁 Models saved to: {os.path.abspath(MODELS_DIR)}')
