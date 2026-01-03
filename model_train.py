import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'crop_data.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'crop_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def preprocess_data(df):
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, scaler, le


def train_model(X_train, y_train, model_type='random_forest'):
    if model_type == 'decision_tree':
        model = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"Model Evaluation Results")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    return accuracy


def save_artifacts(model, scaler, label_encoder):
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    
    print(f"\nArtifacts saved:")
    print(f"  - Model: {MODEL_PATH}")
    print(f"  - Scaler: {SCALER_PATH}")
    print(f"  - Label Encoder: {ENCODER_PATH}")


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, scaler, label_encoder


def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    model, scaler, label_encoder = load_artifacts()
    
    features = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    
    crop_name = label_encoder.inverse_transform(prediction)[0]
    confidence = np.max(probabilities) * 100
    
    top_3_indices = np.argsort(probabilities[0])[-3:][::-1]
    top_3_crops = [
        {
            'crop': label_encoder.inverse_transform([idx])[0],
            'confidence': probabilities[0][idx] * 100
        }
        for idx in top_3_indices
    ]
    
    return {
        'recommended_crop': crop_name,
        'confidence': confidence,
        'top_recommendations': top_3_crops
    }


def main():
    print("CERES Crop Recommendation Model Training")
    print("="*50)
    
    print("\n[1/5] Loading dataset...")
    df = load_data()
    print(f"  - Loaded {len(df)} samples")
    print(f"  - Crops: {df['label'].nunique()} unique types")
    print(f"  - Features: N, P, K, temperature, humidity, ph, rainfall")
    
    print("\n[2/5] Preprocessing data...")
    X, y, scaler, label_encoder = preprocess_data(df)
    print(f"  - Features scaled using StandardScaler")
    print(f"  - Labels encoded: {list(label_encoder.classes_)}")
    
    print("\n[3/5] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Testing samples: {len(X_test)}")
    
    print("\n[4/5] Training Random Forest model...")
    model = train_model(X_train, y_train, model_type='random_forest')
    print("  - Model trained successfully")
    
    print("\n[5/5] Evaluating model...")
    accuracy = evaluate_model(model, X_test, y_test, label_encoder)
    
    save_artifacts(model, scaler, label_encoder)
    
    print("\n" + "="*50)
    print("Testing prediction with sample input:")
    print("="*50)
    test_input = {
        'n': 90, 'p': 42, 'k': 43,
        'temperature': 20.87, 'humidity': 82.0,
        'ph': 6.5, 'rainfall': 202.93
    }
    print(f"Input: {test_input}")
    
    result = predict_crop(**test_input)
    print(f"\nPrediction Results:")
    print(f"  - Recommended Crop: {result['recommended_crop']}")
    print(f"  - Confidence: {result['confidence']:.2f}%")
    print(f"\nTop 3 Recommendations:")
    for i, rec in enumerate(result['top_recommendations'], 1):
        print(f"  {i}. {rec['crop']}: {rec['confidence']:.2f}%")
    
    print("\n" + "="*50)
    print("Model training complete!")
    print("="*50)


if __name__ == '__main__':
    main()
