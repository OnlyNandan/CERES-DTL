"""
Disease Detection Module - Plant Disease Recognition via Image Analysis
Uses TensorFlow/Keras for image classification
"""

import os
import io
import base64
import numpy as np
from PIL import Image
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Disease information database with treatments
DISEASE_DATABASE = {
    'apple_scab': {
        'name': 'Apple Scab',
        'name_hi': 'सेब का पपड़ी रोग',
        'name_kn': 'ಸೇಬು ಸ್ಕ್ಯಾಬ್',
        'severity': 'moderate',
        'symptoms': 'Dark olive-green spots on leaves, velvety texture, leaves may curl',
        'symptoms_hi': 'पत्तियों पर गहरे जैतून-हरे धब्बे, मखमली बनावट, पत्तियां मुड़ सकती हैं',
        'symptoms_kn': 'ಎಲೆಗಳ ಮೇಲೆ ಗಾಢ ಆಲಿವ್-ಹಸಿರು ಕಲೆಗಳು, ವೆಲ್ವೆಟ್ ವಿನ್ಯಾಸ, ಎಲೆಗಳು ಸುರುಳಿಯಾಗಬಹುದು',
        'treatment': [
            'Apply fungicide (Mancozeb/Captan) at 2g/L',
            'Remove and destroy infected leaves',
            'Improve air circulation by pruning',
            'Apply sulfur-based sprays preventively'
        ],
        'treatment_hi': [
            'कवकनाशी (मैनकोजेब/कैप्टन) 2g/L का प्रयोग करें',
            'संक्रमित पत्तियों को हटा दें और नष्ट कर दें',
            'कटाई-छंटाई करके हवा का संचार सुधारें',
            'निवारक रूप से सल्फर-आधारित स्प्रे लागू करें'
        ],
        'treatment_kn': [
            'ಶಿಲೀಂಧ್ರನಾಶಕವನ್ನು (ಮ್ಯಾನ್ಕೋಜೆಬ್/ಕ್ಯಾಪ್ಟನ್) 2g/L ನಲ್ಲಿ ಅನ್ವಯಿಸಿ',
            'ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ನಾಶಮಾಡಿ',
            'ಕತ್ತರಿಸುವ ಮೂಲಕ ಗಾಳಿಯ ಹರಿವನ್ನು ಸುಧಾರಿಸಿ',
            'ಗಂಧಕ ಆಧಾರಿತ ಸ್ಪ್ರೇಗಳನ್ನು ಮುನ್ನೆಚ್ಚರಿಕೆಯಾಗಿ ಅನ್ವಯಿಸಿ'
        ],
        'prevention': 'Plant resistant varieties, maintain good sanitation',
        'prevention_hi': 'प्रतिरोधी किस्में लगाएं, अच्छी स्वच्छता बनाए रखें',
        'prevention_kn': 'ನಿರೋಧಕ ತಳಿಗಳನ್ನು ನೆಡಿ, ಉತ್ತಮ ನೈರ್ಮಲ್ಯವನ್ನು ಕಾಪಾಡಿಕೊಳ್ಳಿ'
    },
    'bacterial_spot': {
        'name': 'Bacterial Spot',
        'name_hi': 'जीवाणु धब्बा',
        'name_kn': 'ಬ್ಯಾಕ್ಟೀರಿಯಾ ಚುಕ್ಕೆ',
        'severity': 'high',
        'symptoms': 'Water-soaked spots, turning brown with yellow halos',
        'symptoms_hi': 'पानी से लथपथ धब्बे, जो पीले घेरे के साथ भूरे हो जाते हैं',
        'symptoms_kn': 'ನೀರಿನಲ್ಲಿ ನೆನೆದ ಕಲೆಗಳು, ಹಳದಿ ಹಾಲೋಗಳೊಂದಿಗೆ ಕಂದು ಬಣ್ಣಕ್ಕೆ ತಿರುಗುತ್ತವೆ',
        'treatment': [
            'Apply copper-based bactericide',
            'Remove infected plant parts',
            'Avoid overhead irrigation',
            'Use disease-free seeds'
        ],
        'treatment_hi': [
            'कॉपर-आधारित जीवाणुनाशक का प्रयोग करें',
            'संक्रमित पौधों के हिस्सों को हटा दें',
            'ऊपरी सिंचाई से बचें',
            'रोग-मुक्त बीजों का उपयोग करें'
        ],
        'treatment_kn': [
            'ತಾಮ್ರ-ಆಧಾರಿತ ಬ್ಯಾಕ್ಟೀರಿಯಾನಾಶಕವನ್ನು ಅನ್ವಯಿಸಿ',
            'ಸೋಂಕಿತ ಸಸ್ಯ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ',
            'ಓವರ್ಹೆಡ್ ನೀರಾವರಿ ತಪ್ಪಿಸಿ',
            'ರೋಗ-ಮುಕ್ತ ಬೀಜಗಳನ್ನು ಬಳಸಿ'
        ],
        'prevention': 'Crop rotation, resistant varieties',
        'prevention_hi': 'फसल चक्र, प्रतिरोधी किस्में',
        'prevention_kn': 'ಬೆಳೆ ಸರದಿ, ನಿರೋಧಕ ತಳಿಗಳು'
    },
    'late_blight': {
        'name': 'Late Blight',
        'name_hi': 'पछेती अंगमारी',
        'name_kn': 'ತಡವಾದ ಬ್ಲೈಟ್',
        'severity': 'severe',
        'symptoms': 'Water-soaked lesions, white mold growth, rapid plant collapse',
        'symptoms_hi': 'पानी से लथपथ घाव, सफेद फफूंद की वृद्धि, पौधे का तेजी से गिरना',
        'symptoms_kn': 'ನೀರಿನಲ್ಲಿ ನೆನೆದ ಗಾಯಗಳು, ಬಿಳಿ ಶಿಲೀಂಧ್ರ ಬೆಳವಣಿಗೆ, ಸಸ್ಯದ ತ್ವರಿತ ಕುಸಿತ',
        'treatment': [
            'Apply Metalaxyl + Mancozeb immediately',
            'Remove and burn infected plants',
            'Improve drainage',
            'Spray every 7-10 days during outbreak'
        ],
        'treatment_hi': [
            'मेटालेक्सिल + मैंकोजेब का तुरंत प्रयोग करें',
            'संक्रमित पौधों को हटा दें और जला दें',
            'जल निकासी में सुधार करें',
            'प्रकोप के दौरान हर 7-10 दिनों में स्प्रे करें'
        ],
        'treatment_kn': [
            'ಮೆಟಾಲಾಕ್ಸಿಲ್ + ಮ್ಯಾನ್ಕೋಜೆಬ್ ಅನ್ನು ತಕ್ಷಣ ಅನ್ವಯಿಸಿ',
            'ಸೋಂಕಿತ ಸಸ್ಯಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಸುಟ್ಟುಹಾಕಿ',
            'ಒಳಚರಂಡಿಯನ್ನು ಸುಧಾರಿಸಿ',
            'ಏಕಾಏಕಿ ಹರಡುವಾಗ ಪ್ರತಿ 7-10 ದಿನಗಳಿಗೊಮ್ಮೆ ಸಿಂಪಡಿಸಿ'
        ],
        'prevention': 'Plant certified disease-free tubers, proper spacing',
        'prevention_hi': 'प्रमाणित रोग-मुक्त कंद रोपें, उचित दूरी',
        'prevention_kn': 'ಪ್ರಮಾಣೀಕೃತ ರೋಗ-ಮುಕ್ತ ಗೆಡ್ಡೆಗಳನ್ನು ನೆಡಿ, ಸರಿಯಾದ ಅಂತರ'
    },
    'early_blight': {
        'name': 'Early Blight',
        'name_hi': 'अगेती अंगमारी',
        'name_kn': 'ಆರಂಭಿಕ ಬ್ಲೈಟ್',
        'severity': 'moderate',
        'symptoms': 'Target-shaped concentric rings on leaves, yellowing',
        'symptoms_hi': 'पत्तियों पर लक्ष्य-आकार के संकेंद्रित छल्ले, पीलापन',
        'symptoms_kn': 'ಎಲೆಗಳ ಮೇಲೆ ಗುರಿ-ಆಕಾರದ ಉಂಗುರಗಳು, ಹಳದಿಯಾಗುವುದು',
        'treatment': [
            'Apply Chlorothalonil or Mancozeb',
            'Remove lower infected leaves',
            'Mulch to prevent soil splash',
            'Maintain adequate nitrogen levels'
        ],
        'treatment_hi': [
            'क्लोरोथालोनिल या मैंकोजेब का प्रयोग करें',
            'निचली संक्रमित पत्तियों को हटा दें',
            'मिट्टी के छींटों को रोकने के लिए मल्चिंग करें',
            'पर्याप्त नाइट्रोजन का स्तर बनाए रखें'
        ],
        'treatment_kn': [
            'ಕ್ಲೋರೊಥಲೋನಿಲ್ ಅಥವಾ ಮ್ಯಾನ್ಕೋಜೆಬ್ ಅನ್ವಯಿಸಿ',
            'ಕೆಳಗಿನ ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ',
            'ಮಣ್ಣು ಚಿಮ್ಮುವುದನ್ನು ತಡೆಯಲು ಹೊದಿಕೆ ಹಾಕಿ',
            'ಸಾಕಷ್ಟು ಸಾರಜನಕ ಮಟ್ಟವನ್ನು ನಿರ್ವಹಿಸಿ'
        ],
        'prevention': 'Crop rotation, stake plants for airflow',
        'prevention_hi': 'फसल चक्र, पौधों को हवा के लिए सहारा दें',
        'prevention_kn': 'ಬೆಳೆ ಸರದಿ, ಗಾಳಿ ಹರಿವಿಗಾಗಿ ಸಸ್ಯಗಳನ್ನು ಬೆಂಬಲಿಸಿ'
    },
    'leaf_mold': {
        'name': 'Leaf Mold',
        'name_hi': 'पत्ती फफूंद',
        'name_kn': 'ಎಲೆ ಶಿಲೀಂಧ್ರ',
        'severity': 'moderate',
        'symptoms': 'Yellow spots on upper leaf, olive-green mold below',
        'symptoms_hi': 'ऊपरी पत्ती पर पीले धब्बे, नीचे जैतून-हरे रंग की फफूंद',
        'symptoms_kn': 'ಎಲೆಗಳ ಮೇಲೆ ಹಳದಿ ಕಲೆಗಳು, ಕೆಳಗೆ ಆಲಿವ್-ಹಸಿರು ಶಿಲೀಂಧ್ರ',
        'treatment': [
            'Improve ventilation',
            'Reduce humidity below 85%',
            'Apply fungicide if severe',
            'Remove infected leaves'
        ],
        'treatment_hi': [
            'हवा का संचार सुधारें',
            'नमी को 85% से कम करें',
            'गंभीर होने पर कवकनाशी का प्रयोग करें',
            'संक्रमित पत्तियों को हटा दें'
        ],
        'treatment_kn': [
            'ಗಾಳಿ ಸಂಪರ್ಕ ಸುಧಾರಿಸಿ',
            'ಆರ್ದ್ರತೆಯನ್ನು 85% ಕ್ಕಿಂತ ಕಡಿಮೆ ಮಾಡಿ',
            'ತೀವ್ರವಾಗಿದ್ದರೆ ಶಿಲೀಂಧ್ರನಾಶಕವನ್ನು ಅನ್ವಯಿಸಿ',
            'ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ'
        ],
        'prevention': 'Proper spacing, avoid overhead watering',
        'prevention_hi': 'उचित दूरी, ऊपरी पानी देने से बचें',
        'prevention_kn': 'ಸರಿಯಾದ ಅಂತರ, ಓವರ್ಹೆಡ್ ನೀರುಹಾಕುವುದನ್ನು ತಪ್ಪಿಸಿ'
    },
    'powdery_mildew': {
        'name': 'Powdery Mildew',
        'name_hi': 'चूर्णी फफूंदी',
        'name_kn': 'ಪುಡಿ ಶಿಲೀಂಧ್ರ',
        'severity': 'moderate',
        'symptoms': 'White powdery coating on leaves, stems, buds',
        'symptoms_hi': 'पत्तियों, तनों, कलियों पर सफेद पाउडर की परत',
        'symptoms_kn': 'ಎಲೆಗಳು, ಕಾಂಡಗಳು, ಮೊಗ್ಗುಗಳ ಮೇಲೆ ಬಿಳಿ ಪುಡಿ ಲೇಪನ',
        'treatment': [
            'Apply sulfur-based fungicide',
            'Spray potassium bicarbonate solution',
            'Neem oil spray (2ml/L)',
            'Remove severely infected parts'
        ],
        'treatment_hi': [
            'सल्फर-आधारित कवकनाशी का प्रयोग करें',
            'पोटेशियम बाइकार्बोनेट घोल का स्प्रे करें',
            'नीम के तेल का स्प्रे (2ml/L)',
            'गंभीर रूप से संक्रमित भागों को हटा दें'
        ],
        'treatment_kn': [
            'ಗಂಧಕ ಆಧಾರಿತ ಶಿಲೀಂಧ್ರನಾಶಕವನ್ನು ಅನ್ವಯಿಸಿ',
            'ಪೊಟ್ಯಾಸಿಯಮ್ ಬೈಕಾರ್ಬನೇಟ್ ದ್ರಾವಣವನ್ನು ಸಿಂಪಡಿಸಿ',
            'ಬೇವು ಎಣ್ಣೆ ಸಿಂಪಡಿಸಿ (2ml/L)',
            'ತೀವ್ರವಾಗಿ ಸೋಂಕಿತ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ'
        ],
        'prevention': 'Adequate spacing, avoid excessive nitrogen',
        'prevention_hi': 'पर्याप्त दूरी, अत्यधिक नाइट्रोजन से बचें',
        'prevention_kn': 'ಸಾಕಷ್ಟು ಅಂತರ, ಅತಿಯಾದ ಸಾರಜನಕವನ್ನು ತಪ್ಪಿಸಿ'
    },
    'rust': {
        'name': 'Rust Disease',
        'name_hi': 'रस्ट रोग',
        'name_kn': 'ತುಕ್ಕು ರೋಗ',
        'severity': 'moderate',
        'symptoms': 'Orange-brown pustules on undersides of leaves',
        'symptoms_hi': 'पत्तियों के नीचे नारंगी-भूरे रंग के दाने',
        'symptoms_kn': 'ಎಲೆಗಳ ಕೆಳಭಾಗದಲ್ಲಿ ಕಿತ್ತಳೆ-ಕಂದು ಬೊಕ್ಕೆಗಳು',
        'treatment': [
            'Apply Propiconazole or Tebuconazole',
            'Remove alternate hosts',
            'Destroy crop residues',
            'Apply at first sign of disease'
        ],
        'treatment_hi': [
            'प्रोपिकोनाज़ोल या टेबुकोनाज़ोल का प्रयोग करें',
            'वैकल्पिक मेजबानों को हटा दें',
            'फसल अवशेषों को नष्ट करें',
            'रोग के पहले संकेत पर प्रयोग करें'
        ],
        'treatment_kn': [
            'ಪ್ರೊಪಿಕೊನಜೋಲ್ ಅಥವಾ ಟೆಬುಕೊನಜೋಲ್ ಅನ್ವಯಿಸಿ',
            'ಪರ್ಯಾಯ ಆತಿಥೇಯರನ್ನು ತೆಗೆದುಹಾಕಿ',
            'ಬೆಳೆ ತ್ಯಾಜ್ಯಗಳನ್ನು ನಾಶಮಾಡಿ',
            'ರೋಗದ ಮೊದಲ ಚಿಹ್ನೆಯಲ್ಲಿ ಅನ್ವಯಿಸಿ'
        ],
        'prevention': 'Resistant varieties, crop rotation',
        'prevention_hi': 'प्रतिरोधी किस्में, फसल चक्र',
        'prevention_kn': 'ನಿರೋಧಕ ತಳಿಗಳು, ಬೆಳೆ ಸರದಿ'
    },
    'septoria_leaf_spot': {
        'name': 'Septoria Leaf Spot',
        'name_hi': 'सेप्टोरिया पत्ती धब्बा',
        'name_kn': 'ಸೆಪ್ಟೋರಿಯಾ ಎಲೆ ಚುಕ್ಕೆ',
        'severity': 'moderate',
        'symptoms': 'Small circular spots with dark borders and tan centers',
        'symptoms_hi': 'काले किनारों और भूरे केंद्रों के साथ छोटे गोलाकार धब्बे',
        'symptoms_kn': 'ಗಾಢ ಗಡಿಗಳು ಮತ್ತು ಕಂದು ಕೇಂದ್ರಗಳನ್ನು ಹೊಂದಿರುವ ಸಣ್ಣ ವೃತ್ತಾಕಾರದ ಕಲೆಗಳು',
        'treatment': [
            'Apply Chlorothalonil fungicide',
            'Remove infected lower leaves',
            'Avoid wetting foliage',
            'Mulch around plants'
        ],
        'treatment_hi': [
            'क्लोरोथालोनिल कवकनाशी का प्रयोग करें',
            'संक्रमित निचली पत्तियों को हटा दें',
            'पत्तियों को गीला करने से बचें',
            'पौधों के चारों ओर मल्चिंग करें'
        ],
        'treatment_kn': [
            'ಕ್ಲೋರೊಥಲೋನಿಲ್ ಶಿಲೀಂಧ್ರನಾಶಕವನ್ನು ಅನ್ವಯಿಸಿ',
            'ಸೋಂಕಿತ ಕೆಳಗಿನ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ',
            'ಎಲೆಗಳನ್ನು ಒದ್ದೆ ಮಾಡುವುದನ್ನು ತಪ್ಪಿಸಿ',
            'ಸಸ್ಯಗಳ ಸುತ್ತಲೂ ಹೊದಿಕೆ ಹಾಕಿ'
        ],
        'prevention': 'Crop rotation, proper plant spacing',
        'prevention_hi': 'फसल चक्र, उचित पौधों की दूरी',
        'prevention_kn': 'ಬೆಳೆ ಸರದಿ, ಸರಿಯಾದ ಸಸ್ಯ ಅಂತರ'
    },
    'yellow_leaf_curl_virus': {
        'name': 'Yellow Leaf Curl Virus',
        'name_hi': 'पीला पत्ता मोड़ विषाणु',
        'name_kn': 'ಹಳದಿ ಎಲೆ ಸುರುಳಿ ವೈರಸ್',
        'severity': 'severe',
        'symptoms': 'Upward curling, yellowing leaves, stunted growth',
        'symptoms_hi': 'ऊपर की ओर मुड़ना, पीली पत्तियां, रुका हुआ विकास',
        'symptoms_kn': 'ಎಲೆಗಳು ಮೇಲಕ್ಕೆ ಸುರುಳಿಯಾಗುವುದು, ಹಳದಿಯಾಗುವುದು, ಬೆಳವಣಿಗೆ ಕುಂಠಿತವಾಗುವುದು',
        'treatment': [
            'No cure - remove infected plants',
            'Control whitefly vectors with Imidacloprid',
            'Use yellow sticky traps',
            'Apply neem oil to control vectors'
        ],
        'treatment_hi': [
            'कोई इलाज नहीं - संक्रमित पौधों को हटा दें',
            'इमिडाक्लोप्रिड के साथ सफेद मक्खी वैक्टर को नियंत्रित करें',
            'पीले चिपचिपे जाल का उपयोग करें',
            'वैक्टर को नियंत्रित करने के लिए नीम का तेल लगाएं'
        ],
        'treatment_kn': [
            'ರೋಗ ಪರಿಹಾರವಿಲ್ಲ - ಸೋಂಕಿತ ಸಸ್ಯಗಳನ್ನು ತೆಗೆದುಹಾಕಿ',
            'ಇಮಿಡಾಕ್ಲೋಪ್ರಿಡ್ ಮೂಲಕ ಬಿಳಿ ನೊಣ ವಾಹಕಗಳನ್ನು ನಿಯಂತ್ರಿಸಿ',
            'ಹಳದಿ ಅಂಟು ಬಲೆಗಳನ್ನು ಬಳಸಿ',
            'ವಾಹಕಗಳನ್ನು ನಿಯಂತ್ರಿಸಲು ಬೇವಿನ ಎಣ್ಣೆ ಅನ್ವಯಿಸಿ'
        ],
        'prevention': 'Resistant varieties, reflective mulches, screen nurseries',
        'prevention_hi': 'प्रतिरोधी किस्में,  नर्सरी को स्क्रीन करें',
        'prevention_kn': 'ನಿರೋಧಕ ತಳಿಗಳು, ಪ್ರತಿಫಲಿತ ಹೊದಿಕೆಗಳು, ನರ್ಸರಿಗಳನ್ನು ರಕ್ಷಿಸಿ'
    },
    'healthy': {
        'name': 'Healthy Plant',
        'name_hi': 'स्वस्थ पौधा',
        'name_kn': 'ಆರೋಗ್ಯಕರ ಸಸ್ಯ',
        'severity': 'none',
        'symptoms': 'No visible disease symptoms',
        'symptoms_hi': 'कोई दिखाई देने वाले रोग के लक्षण नहीं',
        'symptoms_kn': 'ಯಾವುದೇ ಗೋಚರ ರೋಗ ಲಕ್ಷಣಗಳಿಲ್ಲ',
        'treatment': ['Continue regular care and monitoring'],
        'treatment_hi': ['नियमित देखभाल और निगरानी जारी रखें'],
        'treatment_kn': ['ನಿಯಮಿತ ಆರೈಕೆ ಮತ್ತು ಮೇಲ್ವಿಚಾರಣೆಯನ್ನು ಮುಂದುವರಿಸಿ'],
        'prevention': 'Maintain current practices',
        'prevention_hi': 'वर्तमान प्रथाओं को बनाए रखें',
        'prevention_kn': 'ಪ್ರಸ್ತುತ ಅಭ್ಯಾಸಗಳನ್ನು ನಿರ್ವಹಿಸಿ'
    },
    'rice_blast': {
        'name': 'Rice Blast',
        'name_hi': 'धान का ब्लास्ट',
        'name_kn': 'ಭತ್ತದ ಬ್ಲಾಸ್ಟ್',
        'severity': 'severe',
        'symptoms': 'Diamond-shaped lesions on leaves, neck rot',
        'symptoms_hi': 'पत्तियों पर हीरे के आकार के घाव, गर्दन का सड़ना',
        'symptoms_kn': 'ಎಲೆಗಳ ಮೇಲೆ ವಜ್ರದ ಆಕಾರದ ಗಾಯಗಳು, ಕುತ್ತಿಗೆ ಕೊಳೆತ',
        'treatment': [
            'Apply Tricyclazole (0.6g/L) or Isoprothiolane',
            'Drain excess water from fields',
            'Reduce nitrogen application',
            'Apply silicon-based fertilizers'
        ],
        'treatment_hi': [
            'ट्राइसाइक्लाज़ोल (0.6g/L) या आइसोप्रोथियोलेन का प्रयोग करें',
            'खेतों से अतिरिक्त पानी निकाल दें',
            'नाइट्रोजन का प्रयोग कम करें',
            'सिलिकॉन आधारित उर्वरकों का प्रयोग करें'
        ],
        'treatment_kn': [
            '್ರೈಸೈಕ್ಲಾಜೋಲ್ (0.6g/L) ಅಥವಾ ಐಸೊಪ್ರೊಥಿಯೊಲೇನ್ ಅನ್ವಯಿಸಿ',
            'ಹೊಲಗಳಿಂದ ಹೆಚ್ಚುವರಿ ನೀರನ್ನು ಹರಿಸಿ',
            'ಸಾರಜನಕ ಅನ್ವಯಿಕೆ ಕಡಿಮೆ ಮಾಡಿ',
            'ಸಿಲಿಕಾನ್ ಆಧಾರಿತ ರಸಗೊಬ್ಬರಗಳನ್ನು ಅನ್ವಯಿಸಿ'
        ],
        'prevention': 'Resistant varieties, balanced fertilization',
        'prevention_hi': 'प्रतिरोधी किस्में, संतुलित उर्वरक',
        'prevention_kn': 'ನಿರೋಧಕ ತಳಿಗಳು, ಸಮತೋಲಿತ ಫಲೀಕರಣ'
    },
    'brown_spot': {
        'name': 'Brown Spot',
        'name_hi': 'भूरा धब्बा',
        'name_kn': 'ಕಂದು ಚುಕ್ಕೆ',
        'severity': 'moderate',
        'symptoms': 'Brown oval spots with gray center on leaves',
        'symptoms_hi': 'पत्तियों पर ग्रे केंद्र के साथ भूरे अंडाकार धब्बे',
        'symptoms_kn': 'ಎಲೆಗಳ ಮೇಲೆ ಬೂದು ಕೇಂದ್ರವಿರುವ ಕಂದು ಅಂಡಾಕಾರದ ಕಲೆಗಳು',
        'treatment': [
            'Apply Mancozeb or Carbendazim',
            'Improve soil fertility',
            'Proper water management',
            'Seed treatment with fungicide'
        ],
        'treatment_hi': [
            'मैनकोजेब या कार्बेन्डाजिम का प्रयोग करें',
            'मिट्टी की उर्वरता में सुधार करें',
            'उचित जल प्रबंधन',
            'कवकनाशी के साथ बीज उपचार'
        ],
        'treatment_kn': [
            'ಮ್ಯಾನ್ಕೋಜೆಬ್ ಅಥವಾ ಕಾರ್ಬೆಂಡಾಜಿಮ್ ಅನ್ವಯಿಸಿ',
            'ಮಣ್ಣಿನ ಫಲವತ್ತತೆಯನ್ನು ಸುಧಾರಿಸಿ',
            'ಸರಿಯಾದ ನೀರು ನಿರ್ವಹಣೆ',
            'ಶಿಲೀಂಧ್ರನಾಶಕದೊಂದಿಗೆ ಬೀಜ ಸಂಸ್ಕರಣೆ'
        ],
        'prevention': 'Balanced nutrition, proper drainage',
        'prevention_hi': 'संतुलित पोषण, उचित जल निकासी',
        'prevention_kn': 'ಸಮತೋಲಿತ ಪೋಷಣೆ, ಸರಿಯಾದ ಒಳಚರಂಡಿ'
    },
    'bacterial_leaf_blight': {
        'name': 'Bacterial Leaf Blight',
        'name_hi': 'जीवाणु पत्ती झुलसा',
        'name_kn': 'ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಎಲೆ ಬ್ಲೈಟ್',
        'severity': 'severe',
        'symptoms': 'Water-soaked lesions turning yellow-white, wilting',
        'symptoms_hi': 'पानी से लथपथ घाव पीले-सफेद हो जाते हैं, मुरझाना',
        'symptoms_kn': 'ಹಳದಿ-ಬಿಳಿ, ಬಾಡುವ ನೀರಿನಲ್ಲಿ ನೆನೆದ ಗಾಯಗಳು',
        'treatment': [
            'Apply Streptomycin sulfate',
            'Copper hydroxide spray',
            'Drain fields during infection',
            'Remove infected plant debris'
        ],
        'treatment_hi': [
            'स्ट्रेप्टोमाइसिन सल्फेट का प्रयोग करें',
            'कॉपर हाइड्रोक्साइड स्प्रे',
            'संक्रमण के दौरान खेत खाली करें',
            'संक्रमित पौधों के मलबे को हटा दें'
        ],
        'treatment_kn': [
            'ಸ್ಟ್ರೆಪ್ಟೊಮೈಸಿನ್ ಸಲ್ಫೇಟ್ ಅನ್ವಯಿಸಿ',
            'ಕಾಪರ್ ಹೈಡ್ರಾಕ್ಸೈಡ್ ಸ್ಪ್ರೇ',
            'ಸೋಂಕಿನ ಸಮಯದಲ್ಲಿ ಹೊಲಗಳನ್ನು ಬರಿದು ಮಾಡಿ',
            'ಸೋಂಕಿತ ಸಸ್ಯ ಭಗ್ನಾವಶೇಷಗಳನ್ನು ತೆಗೆದುಹಾಕಿ'
        ],
        'prevention': 'Resistant varieties, balanced fertilizer',
        'prevention_hi': 'प्रतिरोधी किस्में, संतुलित उर्वरक',
        'prevention_kn': 'ನಿರೋಧಕ ತಳಿಗಳು, ಸಮತೋಲಿತ ರಸಗೊಬ್ಬರ'
    },
    'cotton_leaf_curl': {
        'name': 'Cotton Leaf Curl Virus',
        'name_hi': 'कपास पत्ती मोड़ रोग',
        'name_kn': 'ಹತ್ತಿ ಎಲೆ ಸುರುಳಿ',
        'severity': 'severe',
        'symptoms': 'Upward/downward leaf curling, vein thickening, stunting',
        'symptoms_hi': 'ऊपर/नीचे पत्ती का मुड़ना, शिरा मोटा होना, बोनापन',
        'symptoms_kn': 'ಎಲೆಗಳು ಮೇಲಕ್ಕೆ/ಕೆಳಕ್ಕೆ ಸುರುಳಿಯಾಗುವುದು, ರಕ್ತನಾಳಗಳ ದಪ್ಪವಾಗುವುದು, ಕುಂಠಿತ',
        'treatment': [
            'Remove and destroy infected plants',
            'Control whitefly with Thiamethoxam',
            'Use neem-based insecticides',
            'Install yellow sticky traps'
        ],
        'treatment_hi': [
            'संक्रमित पौधों को हटा दें और नष्ट कर दें',
            'थियामेथोक्साम के साथ सफेद मक्खी को नियंत्रित करें',
            'नीम आधारित कीटनाशकों का प्रयोग करें',
            'पीले चिपचिपे जाल स्थापित करें'
        ],
        'treatment_kn': [
            'ಸೋಂಕಿತ ಸಸ್ಯಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ನಾಶಮಾಡಿ',
            'ಥಿಯಾಮೆಥಾಕ್ಸಮ್ ಮೂಲಕ ಬಿಳಿ ನೊಣವನ್ನು ನಿಯಂತ್ರಿಸಿ',
            'ಬೇವಿನ ಆಧಾರಿತ ಕೀಟನಾಶಕಗಳನ್ನು ಬಳಸಿ',
            'ಹಳದಿ ಅಂಟು ಬಲೆಗಳನ್ನು ಸ್ಥಾಪಿಸಿ'
        ],
        'prevention': 'Bt cotton varieties, early sowing, vector control',
        'prevention_hi': 'बीटी कपास किस्में, जल्दी बुवाई, वेक्टर नियंत्रण',
        'prevention_kn': 'ಬಿಟಿ ಹತ್ತಿ ತಳಿಗಳು, ಆರಂಭಿಕ ಬಿತ್ತನೆ, ವಾಹಕ ನಿಯಂತ್ರಣ'
    }
}

# Mapping of model class indices to disease keys
CLASS_MAPPING = {
    0: 'apple_scab',
    1: 'bacterial_spot',
    2: 'early_blight',
    3: 'late_blight',
    4: 'leaf_mold',
    5: 'powdery_mildew',
    6: 'rust',
    7: 'septoria_leaf_spot',
    8: 'yellow_leaf_curl_virus',
    9: 'healthy',
    10: 'rice_blast',
    11: 'brown_spot',
    12: 'bacterial_leaf_blight',
    13: 'cotton_leaf_curl'
}


class DiseaseDetector:
    """Plant disease detection using image analysis"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.image_size = (224, 224)
        
    def load_model(self) -> bool:
        """Load the disease detection model"""
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'disease_model.h5')
        
        # Check if TensorFlow model exists
        if os.path.exists(model_path):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path)
                self.model_loaded = True
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        
        # If no TF model, we'll use rule-based detection
        self.model_loaded = False
        return False
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess image for model input"""
        try:
            # Open image from bytes
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize
            image = image.resize(self.image_size)
            
            # Convert to array and normalize
            img_array = np.array(image) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {e}")
    
    def analyze_image_colors(self, image_data: bytes) -> Dict[str, Any]:
        """Analyze image colors for rule-based detection"""
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for analysis
            image = image.resize((100, 100))
            img_array = np.array(image)
            
            # Calculate color statistics
            r_mean = np.mean(img_array[:, :, 0])
            g_mean = np.mean(img_array[:, :, 1])
            b_mean = np.mean(img_array[:, :, 2])
            
            # Calculate yellow/brown indicators
            yellow_ratio = (r_mean + g_mean) / (2 * max(b_mean, 1))
            green_ratio = g_mean / max(r_mean, 1)
            brown_ratio = r_mean / max(g_mean + b_mean, 1)
            
            # Standard deviations for texture analysis
            r_std = np.std(img_array[:, :, 0])
            g_std = np.std(img_array[:, :, 1])
            
            return {
                'r_mean': r_mean,
                'g_mean': g_mean,
                'b_mean': b_mean,
                'yellow_ratio': yellow_ratio,
                'green_ratio': green_ratio,
                'brown_ratio': brown_ratio,
                'r_std': r_std,
                'g_std': g_std,
                'texture_variance': (r_std + g_std) / 2
            }
        except Exception as e:
            return {'error': str(e)}
    
    def detect_disease_rule_based(self, image_data: bytes) -> Dict[str, Any]:
        """Rule-based disease detection when model is not available"""
        colors = self.analyze_image_colors(image_data)
        
        if 'error' in colors:
            return {
                'success': False,
                'error': colors['error']
            }
        
        # Initialize scores for different conditions
        scores = {}
        
        # Extract color metrics
        yellow_ratio = colors['yellow_ratio']
        green_ratio = colors['green_ratio']
        brown_ratio = colors['brown_ratio']
        texture_variance = colors['texture_variance']
        r_mean = colors['r_mean']
        g_mean = colors['g_mean']
        b_mean = colors['b_mean']
        
        # Calculate additional metrics
        brightness = (r_mean + g_mean + b_mean) / 3
        saturation = max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)
        
        # HEALTHY: Strong green, low texture variance, good brightness
        if green_ratio > 1.1 and texture_variance < 30 and saturation > 40:
            scores['healthy'] = 0.7 + (green_ratio - 1.1) * 0.2
        elif green_ratio > 1.0 and texture_variance < 35:
            scores['healthy'] = 0.5
        
        # YELLOW LEAF CURL VIRUS: Very high yellow, curling indicators (unusual aspect)
        if yellow_ratio > 2.0 and green_ratio < 0.9:
            scores['yellow_leaf_curl_virus'] = 0.5 + (yellow_ratio - 2.0) * 0.1
        elif yellow_ratio > 1.8 and texture_variance > 35:
            scores['yellow_leaf_curl_virus'] = 0.45
        
        # EARLY BLIGHT: Target-shaped spots, moderate yellow/brown
        if yellow_ratio > 1.3 and yellow_ratio < 1.8 and texture_variance > 30:
            scores['early_blight'] = 0.55
        
        # LATE BLIGHT: Water-soaked lesions, darker areas
        if brightness < 100 and texture_variance > 40 and brown_ratio > 0.6:
            scores['late_blight'] = 0.6
        elif texture_variance > 45 and green_ratio < 1.0:
            scores['late_blight'] = 0.5
        
        # BROWN SPOT: Obvious brown areas with texture
        if brown_ratio > 0.9 and texture_variance > 25:
            scores['brown_spot'] = 0.65
        elif brown_ratio > 0.7 and texture_variance > 35:
            scores['brown_spot'] = 0.5
        
        # SEPTORIA LEAF SPOT: Small spots, moderate texture
        if texture_variance > 30 and texture_variance < 45 and brown_ratio > 0.5 and brown_ratio < 0.8:
            scores['septoria_leaf_spot'] = 0.55
        
        # RUST: Orange-brown pustules
        if r_mean > g_mean and r_mean > 120 and brown_ratio > 0.6 and brown_ratio < 0.9:
            scores['rust'] = 0.55
        
        # POWDERY MILDEW: Whitish appearance, high brightness
        if brightness > 160 and saturation < 50 and green_ratio < 1.1:
            scores['powdery_mildew'] = 0.6
        elif brightness > 140 and saturation < 40:
            scores['powdery_mildew'] = 0.45
        
        # LEAF MOLD: Yellow spots with darker underside patterns
        if yellow_ratio > 1.2 and yellow_ratio < 1.6 and texture_variance > 25:
            scores['leaf_mold'] = 0.5
        
        # BACTERIAL SPOT: Water-soaked spots with yellow halos
        if texture_variance > 35 and yellow_ratio > 1.1 and green_ratio < 1.0:
            scores['bacterial_spot'] = 0.5
        
        # BACTERIAL LEAF BLIGHT: Similar to bacterial spot but more severe
        if texture_variance > 40 and yellow_ratio > 1.3 and brightness < 130:
            scores['bacterial_leaf_blight'] = 0.55
        
        # RICE BLAST: Diamond-shaped lesions
        if texture_variance > 35 and brown_ratio > 0.5 and green_ratio > 0.7:
            scores['rice_blast'] = 0.45
        
        # If no significant detection, lean towards healthy with moderate confidence
        if not scores or max(scores.values()) < 0.4:
            scores['healthy'] = max(scores.get('healthy', 0), 0.55)
        
        # Get top prediction
        if scores:
            top_disease = max(scores, key=scores.get)
            confidence = min(scores[top_disease] * 100, 85)  # Cap at 85% for rule-based
        else:
            top_disease = 'healthy'
            confidence = 50.0
        
        disease_info = DISEASE_DATABASE.get(top_disease, DISEASE_DATABASE['healthy'])
        
        return {
            'success': True,
            'disease_key': top_disease,
            'disease_name': disease_info['name'],
            'confidence': confidence,
            'severity': disease_info['severity'],
            'symptoms': disease_info['symptoms'],
            'treatment': disease_info['treatment'],
            'prevention': disease_info['prevention'],
            'all_scores': scores,
            'method': 'rule_based',
            'color_analysis': colors
        }
    
    def detect_disease(self, image_data: bytes, language: str = 'en') -> Dict[str, Any]:
        """
        Main disease detection method
        
        Args:
            image_data: Raw image bytes
            language: Language code for response
            
        Returns:
            Detection results with disease info and treatment
        """
        try:
            # Try ML model first
            if self.model_loaded and self.model is not None:
                img_array = self.preprocess_image(image_data)
                predictions = self.model.predict(img_array, verbose=0)
                
                # Get top prediction
                top_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][top_idx]) * 100
                
                disease_key = CLASS_MAPPING.get(top_idx, 'healthy')
                disease_info = DISEASE_DATABASE.get(disease_key, DISEASE_DATABASE['healthy'])
                
                # Get localized fields
                name_key = f'name_{language}' if language != 'en' else 'name'
                disease_name = disease_info.get(name_key, disease_info['name'])
                
                symptoms_key = f'symptoms_{language}' if language != 'en' else 'symptoms'
                treatment_key = f'treatment_{language}' if language != 'en' else 'treatment'
                prevention_key = f'prevention_{language}' if language != 'en' else 'prevention'
                
                # Get top 3 predictions
                top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                alternatives = []
                for idx in top_3_idx[1:]:
                    alt_key = CLASS_MAPPING.get(idx, 'healthy')
                    alt_info = DISEASE_DATABASE.get(alt_key, DISEASE_DATABASE['healthy'])
                    alt_name = alt_info.get(name_key, alt_info['name'])
                    alternatives.append({
                        'disease': alt_name,
                        'confidence': float(predictions[0][idx]) * 100
                    })
                
                return {
                    'success': True,
                    'disease_key': disease_key,
                    'disease_name': disease_name,
                    'confidence': confidence,
                    'severity': disease_info['severity'],
                    'symptoms': disease_info.get(symptoms_key, disease_info['symptoms']),
                    'treatment': disease_info.get(treatment_key, disease_info['treatment']),
                    'prevention': disease_info.get(prevention_key, disease_info['prevention']),
                    'alternatives': alternatives,
                    'method': 'ml_model'
                }
            else:
                # Fallback to rule-based detection
                return self.detect_disease_rule_based(image_data)
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Error analyzing image. Please try again with a clearer image.'
            }
    
    def get_disease_info(self, disease_key: str, language: str = 'en') -> Dict[str, Any]:
        """Get detailed information about a specific disease"""
        if disease_key not in DISEASE_DATABASE:
            return {'success': False, 'error': 'Disease not found'}
        
        info = DISEASE_DATABASE[disease_key]
        name_key = f'name_{language}' if language != 'en' else 'name'
        symptoms_key = f'symptoms_{language}' if language != 'en' else 'symptoms'
        treatment_key = f'treatment_{language}' if language != 'en' else 'treatment'
        prevention_key = f'prevention_{language}' if language != 'en' else 'prevention'
        
        return {
            'success': True,
            'disease_key': disease_key,
            'name': info.get(name_key, info['name']),
            'severity': info['severity'],
            'symptoms': info.get(symptoms_key, info['symptoms']),
            'treatment': info.get(treatment_key, info['treatment']),
            'prevention': info.get(prevention_key, info['prevention'])
        }
    
    def get_all_diseases(self, language: str = 'en') -> list:
        """Get list of all detectable diseases"""
        diseases = []
        name_key = f'name_{language}' if language != 'en' else 'name'
        
        for key, info in DISEASE_DATABASE.items():
            if key != 'healthy':
                diseases.append({
                    'key': key,
                    'name': info.get(name_key, info['name']),
                    'severity': info['severity']
                })
        
        return sorted(diseases, key=lambda x: x['name'])


# Singleton instance
disease_detector = DiseaseDetector()
