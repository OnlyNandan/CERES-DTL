"""
Train Fertilizer Recommendation ML Model
========================================
Creates a Random Forest model for precise NPK fertilizer recommendations
based on crop type, soil conditions, and target yield.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def generate_fertilizer_training_data(n_samples=5000):
    """Generate synthetic but realistic fertilizer recommendation data"""
    
    crops = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'soybean', 'groundnut', 
             'potato', 'tomato', 'onion', 'chickpea', 'pigeon_pea']
    
    # Soil nutrient ranges (current levels in kg/ha)
    soil_n_range = (20, 200)
    soil_p_range = (10, 100)
    soil_k_range = (10, 150)
    soil_ph_range = (4.5, 8.5)
    soil_oc_range = (0.2, 2.0)  # Organic carbon %
    
    # Target yield (quintals/ha)
    target_yield_ranges = {
        'rice': (40, 80), 'wheat': (35, 70), 'maize': (50, 100),
        'cotton': (20, 40), 'sugarcane': (600, 1200), 'soybean': (15, 35),
        'groundnut': (20, 40), 'potato': (200, 400), 'tomato': (300, 600),
        'onion': (200, 400), 'chickpea': (15, 30), 'pigeon_pea': (12, 25)
    }
    
    optimal_npk = {
        'rice': {'N': 120, 'P': 60, 'K': 40},
        'wheat': {'N': 120, 'P': 60, 'K': 40},
        'maize': {'N': 150, 'P': 60, 'K': 40},
        'cotton': {'N': 120, 'P': 50, 'K': 50},
        'sugarcane': {'N': 250, 'P': 115, 'K': 115},
        'soybean': {'N': 30, 'P': 60, 'K': 40},
        'groundnut': {'N': 25, 'P': 50, 'K': 75},
        'potato': {'N': 120, 'P': 60, 'K': 120},
        'tomato': {'N': 150, 'P': 75, 'K': 100},
        'onion': {'N': 100, 'P': 50, 'K': 100},
        'chickpea': {'N': 20, 'P': 60, 'K': 40},
        'pigeon_pea': {'N': 25, 'P': 50, 'K': 30}
    }
    
    data = []
    for _ in range(n_samples):
        crop = np.random.choice(crops)
        
        soil_n = np.random.uniform(*soil_n_range)
        soil_p = np.random.uniform(*soil_p_range)
        soil_k = np.random.uniform(*soil_k_range)
        soil_ph = np.random.uniform(*soil_ph_range)
        soil_oc = np.random.uniform(*soil_oc_range)
        
        yield_range = target_yield_ranges[crop]
        target_yield = np.random.uniform(*yield_range)
        yield_factor = (target_yield - yield_range[0]) / (yield_range[1] - yield_range[0])
        
        
        base_npk = optimal_npk[crop]
        
        n_rec = base_npk['N'] * (0.7 + 0.6 * yield_factor)
        p_rec = base_npk['P'] * (0.7 + 0.6 * yield_factor)
        k_rec = base_npk['K'] * (0.7 + 0.6 * yield_factor)
        
        n_rec = max(0, n_rec - (soil_n * 0.5))  
        p_rec = max(0, p_rec - (soil_p * 0.4))  # 40% availability
        k_rec = max(0, k_rec - (soil_k * 0.4))
        
        # pH adjustment (extreme pH reduces availability)
        if soil_ph < 5.5 or soil_ph > 8.0:
            ph_penalty = 1.15
        elif soil_ph < 6.0 or soil_ph > 7.5:
            ph_penalty = 1.08
        else:
            ph_penalty = 1.0
        
        n_rec *= ph_penalty
        p_rec *= ph_penalty
        k_rec *= ph_penalty
        
        # Organic carbon bonus (higher OC = better nutrient retention = less fertilizer needed)
        oc_factor = max(0.85, 1.0 - (soil_oc - 0.5) * 0.1)
        n_rec *= oc_factor
        
        # Add some random variation (±10%)
        n_rec *= np.random.uniform(0.9, 1.1)
        p_rec *= np.random.uniform(0.9, 1.1)
        k_rec *= np.random.uniform(0.9, 1.1)
        
        data.append({
            'crop': crop,
            'soil_n': round(soil_n, 1),
            'soil_p': round(soil_p, 1),
            'soil_k': round(soil_k, 1),
            'soil_ph': round(soil_ph, 2),
            'soil_oc': round(soil_oc, 2),
            'target_yield': round(target_yield, 1),
            'recommended_n': round(n_rec, 1),
            'recommended_p': round(p_rec, 1),
            'recommended_k': round(k_rec, 1)
        })
    
    return pd.DataFrame(data)

def train_fertilizer_model():
    """Train and save fertilizer recommendation models (separate for N, P, K)"""
    
    print("Generating fertilizer training data...")
    df = generate_fertilizer_training_data(n_samples=5000)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Crops: {df['crop'].unique()}")
    print(f"\nSample data:")
    print(df.head())
    
    # Encode crop names
    label_encoder = LabelEncoder()
    df['crop_encoded'] = label_encoder.fit_transform(df['crop'])
    
    # Features
    feature_cols = ['crop_encoded', 'soil_n', 'soil_p', 'soil_k', 'soil_ph', 'soil_oc', 'target_yield']
    X = df[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train separate models for N, P, K (for better accuracy)
    models = {}
    metrics = {}
    
    for nutrient in ['n', 'p', 'k']:
        print(f"\nTraining model for {nutrient.upper()} recommendation...")
        
        y = df[f'recommended_{nutrient}'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Random Forest Regressor
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"   RMSE: {rmse:.2f} kg/ha")
        print(f"   R² Score: {r2:.4f}")
        
        models[nutrient] = model
        metrics[nutrient] = {'rmse': rmse, 'r2': r2}
    
    # Save models
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    print("\nSaving models...")
    joblib.dump(models, os.path.join(models_dir, 'fertilizer_models.pkl'))
    joblib.dump(label_encoder, os.path.join(models_dir, 'fertilizer_label_encoder.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'fertilizer_scaler.pkl'))
    
    print("Fertilizer recommendation models trained and saved!")
    print(f"\nFinal Metrics:")
    for nutrient, metric in metrics.items():
        print(f"   {nutrient.upper()}: RMSE={metric['rmse']:.2f}, R²={metric['r2']:.4f}")
    
    return models, label_encoder, scaler

if __name__ == '__main__':
    train_fertilizer_model()
