"""
AI Assistant Module - Ollama Integration with Gemma3:12b
Provides conversational AI for farming assistance in multiple languages
"""

import requests
import json
from typing import Generator, Optional, Dict, Any

class OllamaAssistant:
    """AI Assistant using local Ollama with Gemma3:12b model"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "gemma3:12b"
        self.fallback_model = "gemma3:2b"  # Fallback for low-spec devices
        
        # Agricultural context system prompt
        self.system_prompt = """You are CERES AI, an expert agricultural assistant for Indian farmers. You provide advice in the user's preferred language (English, Hindi, Kannada, Telugu, Tamil, or Marathi).

Your expertise includes:
- Crop selection and cultivation practices for Indian climate
- Soil health management and fertilizer recommendations  
- Pest and disease identification and treatment
- Weather-based farming decisions
- Water management and irrigation scheduling
- Market trends and pricing for agricultural commodities
- Government schemes like PM-KISAN, PMFBY, Kisan Credit Card
- Organic farming and sustainable practices
- Regional crop calendars (Kharif, Rabi, Zaid seasons)

Guidelines:
1. Give practical, actionable advice suited to small and marginal farmers
2. Use simple language, avoid technical jargon
3. Consider local conditions and traditional practices
4. Recommend cost-effective solutions
5. Mention government support when relevant
6. Be culturally sensitive and respectful
7. If unsure, suggest consulting local Krishi Vigyan Kendra (KVK)

When user speaks in their regional language, respond in the same language.
Keep responses concise but informative."""

    def check_connection(self) -> Dict[str, Any]:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '').split(':')[0] for m in models]
                
                has_primary = any(self.model.split(':')[0] in name for name in model_names)
                has_fallback = any(self.fallback_model.split(':')[0] in name for name in model_names)
                
                return {
                    'connected': True,
                    'models_available': model_names,
                    'primary_model_available': has_primary,
                    'fallback_model_available': has_fallback,
                    'selected_model': self.model if has_primary else (self.fallback_model if has_fallback else None)
                }
            return {'connected': False, 'error': 'Ollama API not responding'}
        except requests.exceptions.ConnectionError:
            return {'connected': False, 'error': 'Cannot connect to Ollama. Is it running?'}
        except Exception as e:
            return {'connected': False, 'error': str(e)}

    def chat(self, message: str, conversation_history: list = None, 
             language: str = 'en', context: dict = None) -> Dict[str, Any]:
        """
        Send a message to the AI and get a response
        
        Args:
            message: User's message
            conversation_history: Previous messages for context
            language: User's preferred language code
            context: Additional context (weather, location, crop info)
        """
        try:
            # Check connection first
            status = self.check_connection()
            if not status['connected']:
                return {'success': False, 'error': status['error']}
            
            model_to_use = status.get('selected_model', self.model)
            if not model_to_use:
                return {'success': False, 'error': 'No compatible model found. Please install gemma3:12b or gemma3:2b'}
            
            # Build messages array
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add context if provided
            if context:
                context_msg = self._build_context_message(context, language)
                messages.append({"role": "system", "content": context_msg})
            
            # Add conversation history
            if conversation_history:
                for entry in conversation_history[-10:]:  # Keep last 10 messages
                    messages.append({
                        "role": entry.get("role", "user"),
                        "content": entry.get("content", "")
                    })
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Make API call
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model_to_use,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 500
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'response': result.get('message', {}).get('content', ''),
                    'model': model_to_use,
                    'done': result.get('done', True)
                }
            else:
                return {'success': False, 'error': f'API error: {response.status_code}'}
                
        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'Request timed out. The model may be loading.'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def chat_stream(self, message: str, conversation_history: list = None,
                    language: str = 'en', context: dict = None) -> Generator[str, None, None]:
        """Stream response for real-time display"""
        try:
            status = self.check_connection()
            if not status['connected']:
                yield f"Error: {status['error']}"
                return
            
            model_to_use = status.get('selected_model', self.model)
            if not model_to_use:
                yield "Error: No compatible model found"
                return
            
            messages = [{"role": "system", "content": self.system_prompt}]
            
            if context:
                context_msg = self._build_context_message(context, language)
                messages.append({"role": "system", "content": context_msg})
            
            if conversation_history:
                for entry in conversation_history[-10:]:
                    messages.append({
                        "role": entry.get("role", "user"),
                        "content": entry.get("content", "")
                    })
            
            messages.append({"role": "user", "content": message})
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model_to_use,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 500
                    }
                },
                stream=True,
                timeout=120
            )
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'message' in data and 'content' in data['message']:
                            yield data['message']['content']
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            yield f"Error: {str(e)}"

    def _build_context_message(self, context: dict, language: str) -> str:
        """Build context message for AI"""
        parts = [f"Current context for the farmer (Language preference: {language}):"]
        
        if context.get('location'):
            parts.append(f"- Location: {context['location']}")
        
        if context.get('weather'):
            w = context['weather']
            parts.append(f"- Weather: {w.get('temperature', 'N/A')}°C, Humidity: {w.get('humidity', 'N/A')}%")
            if w.get('rainfall'):
                parts.append(f"- Recent rainfall: {w['rainfall']}mm")
        
        if context.get('crop'):
            parts.append(f"- Current crop: {context['crop']}")
        
        if context.get('soil_type'):
            parts.append(f"- Soil type: {context['soil_type']}")
        
        if context.get('farm_size'):
            parts.append(f"- Farm size: {context['farm_size']} hectares")
        
        return "\n".join(parts)

    def get_quick_suggestions(self, language: str = 'en') -> list:
        """Get quick suggestion prompts based on language"""
        suggestions = {
            'en': [
                "What crops should I plant this season?",
                "How to control pests naturally?",
                "When should I irrigate my wheat field?",
                "What government schemes can help me?",
                "How to improve soil fertility?"
            ],
            'hi': [
                "इस मौसम में कौन सी फसल बोऊं?",
                "कीटों को प्राकृतिक तरीके से कैसे नियंत्रित करें?",
                "गेहूं के खेत में सिंचाई कब करें?",
                "कौन सी सरकारी योजनाएं मेरी मदद कर सकती हैं?",
                "मिट्टी की उर्वरता कैसे बढ़ाएं?"
            ],
            'kn': [
                "ಈ ಋತುವಿನಲ್ಲಿ ಯಾವ ಬೆಳೆ ಬೆಳೆಯಬೇಕು?",
                "ಕೀಟಗಳನ್ನು ನೈಸರ್ಗಿಕವಾಗಿ ನಿಯಂತ್ರಿಸುವುದು ಹೇಗೆ?",
                "ನನ್ನ ಗೋಧಿ ಹೊಲಕ್ಕೆ ಯಾವಾಗ ನೀರು ಹಾಕಬೇಕು?",
                "ಯಾವ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು ನನಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
                "ಮಣ್ಣಿನ ಫಲವತ್ತತೆಯನ್ನು ಹೇಗೆ ಸುಧಾರಿಸುವುದು?"
            ],
            'te': [
                "ఈ సీజన్‌లో ఏ పంట వేయాలి?",
                "చీడపీడలను సహజంగా నియంత్రించడం ఎలా?",
                "నా గోధుమ పొలానికి ఎప్పుడు నీరు పెట్టాలి?",
                "ఏ ప్రభుత్వ పథకాలు నాకు సహాయం చేయగలవు?",
                "నేల సారాన్ని ఎలా మెరుగుపరచాలి?"
            ],
            'ta': [
                "இந்த பருவத்தில் எந்த பயிர் நடவு செய்ய வேண்டும்?",
                "பூச்சிகளை இயற்கையாக கட்டுப்படுத்துவது எப்படி?",
                "கோதுமை வயலுக்கு எப்போது நீர் பாய்ச்ச வேண்டும்?",
                "எந்த அரசு திட்டங்கள் எனக்கு உதவும்?",
                "மண் வளத்தை மேம்படுத்துவது எப்படி?"
            ],
            'mr': [
                "या हंगामात कोणते पीक घ्यावे?",
                "कीटकांचे नैसर्गिक नियंत्रण कसे करावे?",
                "गहू शेताला पाणी कधी द्यावे?",
                "कोणत्या सरकारी योजना मला मदत करू शकतात?",
                "जमिनीची सुपीकता कशी वाढवावी?"
            ]
        }
        return suggestions.get(language, suggestions['en'])


# Singleton instance
ai_assistant = OllamaAssistant()
