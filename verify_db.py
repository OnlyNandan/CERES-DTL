from disease_detector import DISEASE_DATABASE
import sys

try:
    info = DISEASE_DATABASE['early_blight']
    print(f"Name: {info.get('name')}")
    print(f"Name KN: {info.get('name_kn')}")
    print(f"Symptoms KN: {info.get('symptoms_kn')}")
    
    if 'symptoms_kn' in info:
        print("symptoms_kn FOUND")
    else:
        print("symptoms_kn MISSING")

except Exception as e:
    print(f"Error: {e}")
