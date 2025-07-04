import requests
import time
import threading
from datetime import datetime, timedelta
import json
import pyotp
from env_manager import EnvManager

class AngelOneAPI:
    def __init__(self):
        self.env_manager = EnvManager()
        self.base_url = "https://apiconnect.angelbroking.com"
        self.websocket_url = "wss://smartapisocket.angelone.in/smart-stream"
        
        # Load credentials from env
        self.api_key = self.env_manager.get_env('ANGEL_API_KEY')
        self.client_id = self.env_manager.get_env('ANGEL_CLIENT_ID')
        self.pin = self.env_manager.get_env('ANGEL_PIN')
        self.totp_secret = self.env_manager.get_env('ANGEL_TOTP_SECRET')
        
        # JWT management
        self.jwt_token = self.env_manager.get_env('ANGEL_JWT_TOKEN')
        self.refresh_token = self.env_manager.get_env('ANGEL_REFRESH_TOKEN')
        self.feed_token = self.env_manager.get_env('ANGEL_FEED_TOKEN')
        self.token_expiry = self.env_manager.get_env('ANGEL_TOKEN_EXPIRY')
        
        # Auto refresh setup
        self.auto_refresh_thread = None
        self.stop_refresh = False
        
        if self.jwt_token and self.is_token_valid():
            self.start_auto_refresh()
    
    def generate_totp(self):
        """Generate TOTP for 2FA with detailed error messages"""
        if not self.totp_secret:
            print("[TOTP ERROR] No TOTP secret found. Please set ANGEL_TOTP_SECRET in your environment.")
            return None
        try:
            if len(self.totp_secret.strip()) < 16:
                print(f"[TOTP ERROR] TOTP secret too short: '{self.totp_secret}'. It should be at least 16 characters (base32). Check for missing/extra characters.")
                return None
            totp = pyotp.TOTP(self.totp_secret)
            code = totp.now()
            if not code or len(code) != 6:
                print(f"[TOTP ERROR] Generated TOTP code is invalid: {code}")
                return None
            return code
        except Exception as e:
            print(f"[TOTP ERROR] Exception during TOTP generation: {e}. Secret used: '{self.totp_secret}'")
            return None
    
    def login(self):
        """Login to Angel One API with complete flow"""
        try:
            # Generate TOTP
            totp_code = self.generate_totp()
            if not totp_code:
                return {"success": False, "message": "TOTP generation failed"}
            
            login_url = f"{self.base_url}/rest/auth/angelbroking/user/v1/loginByPassword"
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "192.168.1.1",
                "X-ClientPublicIP": "106.193.147.98",
                "X-MACAddress": "fe80::216:3eff:fe00:362",
                "X-PrivateKey": self.api_key
            }
            
            data = {
                "clientcode": self.client_id,
                "password": self.pin,
                "totp": totp_code
            }
            
            response = requests.post(login_url, json=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status'):
                    # Save tokens
                    self.jwt_token = result['data']['jwtToken']
                    self.refresh_token = result['data']['refreshToken']
                    self.feed_token = result['data']['feedToken']
                    
                    # Calculate expiry (JWT typically expires in 1 hour)
                    expiry_time = datetime.now() + timedelta(hours=1)
                    self.token_expiry = expiry_time.isoformat()
                    
                    # Save to env
                    self.save_tokens_to_env()
                    
                    # Start auto refresh
                    self.start_auto_refresh()
                    
                    return {
                        "success": True, 
                        "message": "Login successful",
                        "data": {
                            "jwtToken": self.jwt_token,
                            "feedToken": self.feed_token,
                            "refreshToken": self.refresh_token
                        }
                    }
                else:
                    return {"success": False, "message": result.get('message', 'Login failed')}
            else:
                return {"success": False, "message": f"HTTP Error: {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "message": f"Login error: {str(e)}"}
    
    def refresh_jwt_token(self):
        """Refresh JWT token using refresh token"""
        try:
            refresh_url = f"{self.base_url}/rest/auth/angelbroking/jwt/v1/generateTokens"
            
            headers = {
                "Authorization": f"Bearer {self.jwt_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "192.168.1.1",
                "X-ClientPublicIP": "106.193.147.98",
                "X-MACAddress": "fe80::216:3eff:fe00:362",
                "X-PrivateKey": self.api_key
            }
            
            data = {
                "refreshToken": self.refresh_token
            }
            
            response = requests.post(refresh_url, json=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status'):
                    # Update tokens
                    self.jwt_token = result['data']['jwtToken']
                    self.refresh_token = result['data']['refreshToken']
                    
                    # Update expiry
                    expiry_time = datetime.now() + timedelta(hours=1)
                    self.token_expiry = expiry_time.isoformat()
                    
                    # Save to env
                    self.save_tokens_to_env()
                    
                    return {"success": True, "message": "Token refreshed successfully"}
                else:
                    return {"success": False, "message": result.get('message', 'Token refresh failed')}
            else:
                return {"success": False, "message": f"HTTP Error: {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "message": f"Token refresh error: {str(e)}"}
    
    def is_token_valid(self):
        """Check if current JWT token is valid"""
        if not self.jwt_token or not self.token_expiry:
            return False
        
        try:
            expiry_dt = datetime.fromisoformat(self.token_expiry)
            # Check if token expires in next 5 minutes
            return datetime.now() < (expiry_dt - timedelta(minutes=5))
        except:
            return False
    
    def start_auto_refresh(self):
        """Start automatic token refresh every 30 seconds"""
        if self.auto_refresh_thread and self.auto_refresh_thread.is_alive():
            return
        
        self.stop_refresh = False
        self.auto_refresh_thread = threading.Thread(target=self._auto_refresh_worker)
        self.auto_refresh_thread.daemon = True
        self.auto_refresh_thread.start()
    
    def _auto_refresh_worker(self):
        """Background worker for auto-refreshing tokens"""
        while not self.stop_refresh:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                if not self.is_token_valid():
                    print("Token expired or expiring soon, refreshing...")
                    result = self.refresh_jwt_token()
                    if result['success']:
                        print("Token refreshed successfully")
                    else:
                        print(f"Token refresh failed: {result['message']}")
                        # Try re-login if refresh fails
                        login_result = self.login()
                        if login_result['success']:
                            print("Re-login successful")
                        else:
                            print(f"Re-login failed: {login_result['message']}")
                            
            except Exception as e:
                print(f"Auto refresh error: {e}")
    
    def stop_auto_refresh(self):
        """Stop automatic token refresh"""
        self.stop_refresh = True
        if self.auto_refresh_thread:
            self.auto_refresh_thread.join(timeout=1)
    
    def save_tokens_to_env(self):
        """Save tokens to environment file"""
        self.env_manager.set_env('ANGEL_JWT_TOKEN', self.jwt_token)
        self.env_manager.set_env('ANGEL_REFRESH_TOKEN', self.refresh_token)
        self.env_manager.set_env('ANGEL_FEED_TOKEN', self.feed_token)
        self.env_manager.set_env('ANGEL_TOKEN_EXPIRY', self.token_expiry)
    
    def save_credentials_to_env(self, api_key, client_id, pin, totp_secret):
        """Save credentials to environment file"""
        self.env_manager.set_env('ANGEL_API_KEY', api_key)
        self.env_manager.set_env('ANGEL_CLIENT_ID', client_id)
        self.env_manager.set_env('ANGEL_PIN', pin)
        self.env_manager.set_env('ANGEL_TOTP_SECRET', totp_secret)
        
        # Update instance variables
        self.api_key = api_key
        self.client_id = client_id
        self.pin = pin
        self.totp_secret = totp_secret
    
    def get_profile(self):
        """Get user profile to verify connection"""
        try:
            if not self.is_token_valid():
                refresh_result = self.refresh_jwt_token()
                if not refresh_result['success']:
                    return {"success": False, "message": "Token refresh failed"}
            
            url = f"{self.base_url}/rest/secure/angelbroking/user/v1/getProfile"
            
            headers = {
                "Authorization": f"Bearer {self.jwt_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "192.168.1.1",
                "X-ClientPublicIP": "106.193.147.98",
                "X-MACAddress": "fe80::216:3eff:fe00:362",
                "X-PrivateKey": self.api_key
            }
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status'):
                    return {"success": True, "data": result['data']}
                else:
                    return {"success": False, "message": result.get('message', 'Profile fetch failed')}
            else:
                return {"success": False, "message": f"HTTP Error: {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "message": f"Profile fetch error: {str(e)}"}
    
    def get_ltp(self, exchange, trading_symbol, symbol_token):
        """Get Last Traded Price"""
        try:
            if not self.is_token_valid():
                refresh_result = self.refresh_jwt_token()
                if not refresh_result['success']:
                    return {"success": False, "message": "Token refresh failed"}
            
            url = f"{self.base_url}/rest/secure/angelbroking/order/v1/getLTP"
            
            headers = {
                "Authorization": f"Bearer {self.jwt_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "192.168.1.1",
                "X-ClientPublicIP": "106.193.147.98",
                "X-MACAddress": "fe80::216:3eff:fe00:362",
                "X-PrivateKey": self.api_key
            }
            
            data = {
                "exchange": exchange,
                "tradingsymbol": trading_symbol,
                "symboltoken": symbol_token
            }
            
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status'):
                    return {"success": True, "data": result['data']}
                else:
                    return {"success": False, "message": result.get('message', 'LTP fetch failed')}
            else:
                return {"success": False, "message": f"HTTP Error: {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "message": f"LTP fetch error: {str(e)}"}
