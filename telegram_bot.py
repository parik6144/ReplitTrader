
import requests
import json
from datetime import datetime
from env_manager import EnvManager

class TelegramBot:
    def __init__(self):
        self.env_manager = EnvManager()
        self.bot_token = self.env_manager.get_env('TELEGRAM_BOT_TOKEN')
        self.chat_id = self.env_manager.get_env('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None
    
    def test_connection(self):
        """Test Telegram bot connection"""
        try:
            if not self.bot_token or not self.chat_id:
                return {
                    'success': False,
                    'message': 'Bot token or chat ID not configured'
                }
            
            # Test with getMe
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                bot_info = response.json()
                
                # Test sending message
                test_msg = f"✅ Bot Connection Test\nTime: {datetime.now().strftime('%H:%M:%S')}\nBot is working!"
                send_result = self.send_message(test_msg)
                
                if send_result:
                    return {
                        'success': True,
                        'message': 'Bot connected and test message sent',
                        'chat_info': bot_info
                    }
                else:
                    return {
                        'success': False,
                        'message': 'Bot connected but failed to send message'
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
    
    def send_message(self, message, parse_mode='HTML'):
        """Send message to Telegram"""
        try:
            if not self.base_url or not self.chat_id:
                return False
            
            url = f"{self.base_url}/sendMessage"
            
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, json=data, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            print(f"Telegram send error: {e}")
            return False
    
    def send_signal_alert(self, signal_data):
        """Send trading signal alert"""
        try:
            message = f"""
🚨 <b>TRADING SIGNAL</b> 🚨

📊 <b>Symbol:</b> {signal_data.get('symbol', 'N/A')}
📈 <b>Signal:</b> {signal_data.get('signal', 'N/A')}
💰 <b>Entry:</b> {signal_data.get('entry', 'N/A')}
🎯 <b>Target:</b> {signal_data.get('target', 'N/A')}
🛑 <b>Stop Loss:</b> {signal_data.get('sl', 'N/A')}
📊 <b>Confidence:</b> {signal_data.get('confidence', 'N/A')}%

⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}

<i>Trade responsibly! 📈</i>
            """
            
            return self.send_message(message)
            
        except Exception as e:
            print(f"Signal alert error: {e}")
            return False
    
    def send_market_update(self, update_data):
        """Send market update"""
        try:
            message = f"""
📈 <b>MARKET UPDATE</b>

🎯 <b>NIFTY:</b> {update_data.get('nifty', 'N/A')}
🏦 <b>BANKNIFTY:</b> {update_data.get('banknifty', 'N/A')}
🌍 <b>Global Sentiment:</b> {update_data.get('sentiment', 'N/A')}

⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}

<i>Stay updated! 📊</i>
            """
            
            return self.send_message(message)
            
        except Exception as e:
            print(f"Market update error: {e}")
            return False
