import os
import requests
import json
from datetime import datetime
from env_manager import EnvManager

class GeminiAPI:
    def __init__(self):
        self.env_manager = EnvManager()
        self.api_key = self.env_manager.get_env('GEMINI_API_KEY')
        self.base_url = "https://generativelanguage.googleapis.com/v1"
        
    def test_connection(self):
        """Test Gemini API connection"""
        try:
            if not self.api_key:
                return {
                    'success': False, 
                    'message': 'No API key found'
                }
            
            url = f"{self.base_url}/models/gemini-1.5-pro:generateContent"
            
            headers = {
                'Content-Type': 'application/json',
                'x-goog-api-key': self.api_key
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": "Hello, test connection. Reply with just 'Connection successful'"
                    }]
                }]
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['candidates'][0]['content']['parts'][0]['text']
                return {
                    'success': True,
                    'message': 'Connection successful',
                    'response': ai_response
                }
            else:
                return {
                    'success': False,
                    'message': f'HTTP {response.status_code}: {response.text}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Connection error: {str(e)}'
            }
    
    def generate_signal_analysis(self, market_data):
        """Generate AI analysis for market signals"""
        try:
            if not self.api_key:
                return "⚠️ Gemini API key not configured"
            
            prompt = f"""
            आप एक professional Indian stock market analyst हैं। 
            नीचे दिए गए market data का विश्लेषण करें:
            
            Market Data: {json.dumps(market_data, indent=2)}
            
            कृपया निम्नलिखित format में analysis दें:
            1. Market Sentiment (Bullish/Bearish/Neutral)
            2. Key Levels (Support/Resistance)
            3. Trading Recommendation
            4. Risk Assessment
            5. Time Frame Analysis
            
            Hindi और English दोनों में जवाब दें।
            """
            
            url = f"{self.base_url}/models/gemini-1.5-pro:generateContent"
            
            headers = {
                'Content-Type': 'application/json',
                'x-goog-api-key': self.api_key
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return f"⚠️ AI Analysis failed: {response.text}"
                
        except Exception as e:
            return f"⚠️ AI Analysis error: {str(e)}"
    
    def generate_market_insight(self, segment_analyses, global_sentiment):
        """Generate market insights from analysis"""
        try:
            if not self.api_key:
                return "AI Analysis unavailable - Configure Gemini API"
            
            prompt = f"""
            Market Analysis Summary:
            Segments: {json.dumps(segment_analyses, indent=2)}
            Global Sentiment: {json.dumps(global_sentiment, indent=2)}
            
            Generate a concise market insight in Hindi and English (2-3 sentences max).
            Include:
            1. Overall market direction
            2. Best opportunity segment
            3. Risk level
            
            Keep it professional and actionable.
            """
            
            url = f"{self.base_url}/models/gemini-1.5-pro:generateContent"
            
            headers = {
                'Content-Type': 'application/json',
                'x-goog-api-key': self.api_key
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=20)
            
            if response.status_code == 200:
                result = response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "Market में volatility देखी जा रही है। Careful trading की सलाह।"
                
        except Exception as e:
            return f"AI insight generation error: {str(e)}"
