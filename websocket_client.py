import websocket
import threading
import json

# Global dict to store latest market data for each symbol
live_market_data = {}
live_market_data_lock = threading.Lock()

class AngelWebSocketClient:
    def __init__(self, feed_token, client_code):
        self.ws_url = f"wss://smartapisocket.angelone.in/smart-stream"
        self.feed_token = feed_token
        self.client_code = client_code
        self.ws = None

    def connect(self):
        headers = {
            "Authorization": f"Bearer {self.feed_token}",
            "x-client-code": self.client_code,
            "x-feed-token": self.feed_token
        }

        self.ws = websocket.WebSocketApp(
            self.ws_url,
            header=headers,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        threading.Thread(target=self.ws.run_forever).start()

    def on_open(self, ws):
        print("WebSocket connection opened.")

    def on_message(self, ws, message):
        data = json.loads(message)
        # You may need to adjust the key depending on Angel's message format
        symbol = data.get('token') or data.get('symbol')
        if symbol:
            with live_market_data_lock:
                live_market_data[symbol] = data
        print("Received:", data)

    def on_error(self, ws, error):
        print("WebSocket error:", error)

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket closed:", close_status_code, close_msg)

    def subscribe(self, tokens):
        for token in tokens:
            request = {
                "action": "subscribe",
                "params": {
                    "mode": "FULL",
                    "token": token
                }
            }
            self.ws.send(json.dumps(request))