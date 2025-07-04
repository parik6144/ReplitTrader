import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Optional
from gemini_api import GeminiAPI
from database_manager import DatabaseManager

class AdvancedMarketAnalyzer:
    def __init__(self):
        self.gemini_api = GeminiAPI()
        self.db_manager = DatabaseManager()

        # Segment mapping
        self.segments = {
            'NIFTY': {
                'symbol': '^NSEI',
                'option_symbol': 'NIFTY',
                'lot_size': 25,
                'strike_gap': 50
            },
            'BANKNIFTY': {
                'symbol': '^NSEBANK',
                'option_symbol': 'BANKNIFTY',
                'lot_size': 15,
                'strike_gap': 100
            },
            'FINNIFTY': {
                'symbol': 'NIFTY_FIN_SERVICE.NS',
                'option_symbol': 'FINNIFTY',
                'lot_size': 25,
                'strike_gap': 50
            }
        }

        # Key stocks for analysis
        self.key_stocks = {
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS',
            'HDFCBANK': 'HDFCBANK.NS',
            'INFY': 'INFY.NS',
            'ICICIBANK': 'ICICIBANK.NS',
            'LT': 'LT.NS',
            'ITC': 'ITC.NS',
            'WIPRO': 'WIPRO.NS',
            'MARUTI': 'MARUTI.NS',
            'BAJFINANCE': 'BAJFINANCE.NS'
        }

        self.symbols_map = {
            "NIFTY": "^NSEI",
            "BANKNIFTY": "^NSEBANK",
            "FINNIFTY": "^CNXFIN",
            "RELIANCE": "RELIANCE.NS",
            "TCS": "TCS.NS",
            "HDFC": "HDFCBANK.NS",
            "INFOSYS": "INFY.NS",
            "ITC": "ITC.NS"
        }

    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate RSI, MACD, EMA and other indicators"""
        try:
            if data.empty or len(data) < 14:
                return {
                    'rsi': 50.0,
                    'macd': 0.0,
                    'macd_signal': 0.0,
                    'macd_histogram': 0.0,
                    'ema_12': 0.0,
                    'ema_26': 0.0,
                    'volatility': 0.0,
                    'support': 0.0,
                    'resistance': 0.0,
                    'current_price': 0.0,
                    'volume': 0
                }

            # Ensure we have Close column
            if 'Close' not in data.columns:
                return {
                    'rsi': 50.0,
                    'macd': 0.0,
                    'macd_signal': 0.0,
                    'macd_histogram': 0.0,
                    'ema_12': 0.0,
                    'ema_26': 0.0,
                    'volatility': 0.0,
                    'support': 0.0,
                    'resistance': 0.0,
                    'current_price': 0.0,
                    'volume': 0
                }

            # RSI Calculation
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()

            # Avoid division by zero
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

            # EMA Calculation
            ema_12 = data['Close'].ewm(span=12, min_periods=1).mean()
            ema_26 = data['Close'].ewm(span=26, min_periods=1).mean()

            # MACD
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9, min_periods=1).mean()
            macd_histogram = macd - macd_signal

            # Volatility (using available data)
            min_periods = min(20, len(data))
            volatility = data['Close'].pct_change().rolling(window=min_periods, min_periods=1).std() * np.sqrt(252) * 100

            # Support and Resistance
            period = min(20, len(data))
            recent_high = data['High'].rolling(window=period, min_periods=1).max().iloc[-1]
            recent_low = data['Low'].rolling(window=period, min_periods=1).min().iloc[-1]

            # Get last valid values
            rsi_value = rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0
            macd_value = macd.iloc[-1] if not macd.empty and not pd.isna(macd.iloc[-1]) else 0.0
            macd_signal_value = macd_signal.iloc[-1] if not macd_signal.empty and not pd.isna(macd_signal.iloc[-1]) else 0.0
            macd_histogram_value = macd_histogram.iloc[-1] if not macd_histogram.empty and not pd.isna(macd_histogram.iloc[-1]) else 0.0
            ema_12_value = ema_12.iloc[-1] if not ema_12.empty and not pd.isna(ema_12.iloc[-1]) else 0.0
            ema_26_value = ema_26.iloc[-1] if not ema_26.empty and not pd.isna(ema_26.iloc[-1]) else 0.0
            volatility_value = volatility.iloc[-1] if not volatility.empty and not pd.isna(volatility.iloc[-1]) else 0.0

            return {
                'rsi': float(rsi_value),
                'macd': float(macd_value),
                'macd_signal': float(macd_signal_value),
                'macd_histogram': float(macd_histogram_value),
                'ema_12': float(ema_12_value),
                'ema_26': float(ema_26_value),
                'volatility': float(volatility_value),
                'support': float(recent_low) if not pd.isna(recent_low) else 0.0,
                'resistance': float(recent_high) if not pd.isna(recent_high) else 0.0,
                'current_price': float(data['Close'].iloc[-1]) if not pd.isna(data['Close'].iloc[-1]) else 0.0,
                'volume': int(data['Volume'].iloc[-1]) if 'Volume' in data.columns and not pd.isna(data['Volume'].iloc[-1]) else 0
            }

        except Exception as e:
            print(f"Technical indicators calculation error: {e}")
            return {
                'rsi': 50.0,
                'macd': 0.0,
                'macd_signal': 0.0,
                'macd_histogram': 0.0,
                'ema_12': 0.0,
                'ema_26': 0.0,
                'volatility': 0.0,
                'support': 0.0,
                'resistance': 0.0,
                'current_price': 0.0,
                'volume': 0
            }

    def get_technical_indicators(self, symbol, period="1mo"):
        """Get technical indicators for a symbol"""
        try:
            yf_symbol = self.symbols_map.get(symbol, f"{symbol}.NS")
            ticker = yf.Ticker(yf_symbol)
            
            # Add timeout to prevent hanging
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError("Data fetch timeout")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)  # 10 second timeout
            
            try:
                hist = ticker.history(period=period)
            finally:
                signal.alarm(0)  # Cancel alarm

            if hist.empty:
                return None

            # Calculate indicators
            close_prices = hist['Close']
            high_prices = hist['High']
            low_prices = hist['Low']

            # RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # MACD
            exp1 = close_prices.ewm(span=12).mean()
            exp2 = close_prices.ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()

            # Moving averages
            sma_20 = close_prices.rolling(window=20).mean()
            sma_50 = close_prices.rolling(window=50).mean()

            # Volatility
            volatility = close_prices.pct_change().rolling(window=20).std() * np.sqrt(252) * 100

            current_price = close_prices.iloc[-1]

            return {
                'current_price': current_price,
                'rsi': rsi.iloc[-1],
                'macd': macd.iloc[-1],
                'signal': signal.iloc[-1],
                'sma_20': sma_20.iloc[-1],
                'sma_50': sma_50.iloc[-1],
                'volatility': volatility.iloc[-1],
                'high_52w': high_prices.max(),
                'low_52w': low_prices.min(),
                'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
            }

        except Exception as e:
            print(f"Technical indicators error for {symbol}: {e}")
            return None

    def analyze_segment(self, segment):
        """Analyze a market segment"""
        try:
            indicators = self.get_technical_indicators(segment)
            if not indicators:
                return None

            # Calculate strength score
            strength_score = 0

            # RSI analysis
            if indicators['rsi'] < 30:
                strength_score += 25  # Oversold
            elif indicators['rsi'] < 50:
                strength_score += 15
            elif indicators['rsi'] < 70:
                strength_score += 10
            else:
                strength_score += 5  # Overbought

            # MACD analysis
            if indicators['macd'] > indicators['signal']:
                strength_score += 20
            else:
                strength_score += 5

            # Price vs Moving averages
            if indicators['current_price'] > indicators['sma_20']:
                strength_score += 15
            if indicators['current_price'] > indicators['sma_50']:
                strength_score += 10

            # Volatility analysis
            if indicators['volatility'] < 20:
                strength_score += 10
            elif indicators['volatility'] < 30:
                strength_score += 5

            # Momentum score
            momentum_score = 0
            price_change = ((indicators['current_price'] - indicators['sma_20']) / indicators['sma_20']) * 100

            if price_change > 2:
                momentum_score += 30
            elif price_change > 1:
                momentum_score += 20
            elif price_change > 0:
                momentum_score += 10
            else:
                momentum_score += 5

            # Volume analysis (if available)
            if indicators['volume'] > 0:
                momentum_score += 10

            # Generate recommendation
            total_score = strength_score + momentum_score

            if total_score > 160:
                action = "STRONG BUY"
                confidence = 85
            elif total_score > 120:
                action = "BUY"
                confidence = 75
            elif total_score > 80:
                action = "HOLD"
                confidence = 65
            else:
                action = "AVOID"
                confidence = 60

            # Determine option type and strike
            if indicators['rsi'] < 50:
                option_type = "CE"
                strike_adjustment = 1.02
            else:
                option_type = "PE"
                strike_adjustment = 0.98

            suggested_strike = indicators['current_price'] * strike_adjustment

            return {
                'segment': segment,
                'indicators': indicators,
                'strength_score': strength_score,
                'momentum_score': momentum_score,
                'recommendation': {
                    'action': action,
                    'confidence': confidence,
                    'option_type': option_type,
                    'suggested_strike': suggested_strike,
                    'current_price': indicators['current_price'],
                    'target_multiplier': 1.05 if action in ['BUY', 'STRONG BUY'] else 0.95,
                    'lot_size': 50,  # Default lot size
                    'reasoning': f"RSI: {indicators['rsi']:.1f}, MACD: {indicators['macd']:.2f}, Strength: {strength_score}/100"
                }
            }

        except Exception as e:
            print(f"Segment analysis error for {segment}: {e}")
            return None

    def analyze_all_segments(self):
        """Analyze all market segments"""
        segments = ["NIFTY", "BANKNIFTY", "FINNIFTY", "RELIANCE", "TCS"]
        analyses = []

        for segment in segments:
            analysis = self.analyze_segment(segment)
            if analysis:
                analyses.append(analysis)

        # Sort by total score
        analyses.sort(key=lambda x: x['strength_score'] + x['momentum_score'], reverse=True)

        return analyses

    def analyze_segment_strength(self, segment: str) -> Dict:
        """Analyze individual segment strength and momentum"""
        try:
            symbol = self.segments[segment]['symbol']

            # Get data for different timeframes with error handling
            try:
                data_1d = yf.download(symbol, period="5d", interval="1d", auto_adjust=True)
            except Exception as e:
                print(f"Error downloading 1d data for {segment}: {e}")
                data_1d = pd.DataFrame()

            try:
                data_1h = yf.download(symbol, period="5d", interval="1h", auto_adjust=True)
            except Exception as e:
                print(f"Error downloading 1h data for {segment}: {e}")
                data_1h = pd.DataFrame()

            try:
                data_15m = yf.download(symbol, period="1d", interval="15m", auto_adjust=True)
            except Exception as e:
                print(f"Error downloading 15m data for {segment}: {e}")
                data_15m = pd.DataFrame()

            # Check if we have any valid data
            if data_1d.empty and data_1h.empty and data_15m.empty:
                print(f"No data available for {segment}")
                return None

            # Use the best available data
            primary_data = data_1d if not data_1d.empty else (data_1h if not data_1h.empty else data_15m)

            if primary_data.empty:
                return None

            # Calculate indicators for available timeframes
            indicators_1d = self.calculate_technical_indicators(primary_data)

            # Calculate strength score (0-100)
            strength_score = 0
            momentum_score = 0

            # RSI component (30%)
            rsi = indicators_1d['rsi']
            if 40 < rsi < 60:
                strength_score += 30
            elif 30 < rsi < 70:
                strength_score += 20
            elif rsi > 70:
                strength_score += 10  # Overbought
            else:
                strength_score += 5   # Oversold

            # MACD component (25%)
            if indicators_1d['macd'] > indicators_1d['macd_signal']:
                strength_score += 25
                momentum_score += 30

            # EMA crossover (20%)
            if indicators_1d['ema_12'] > indicators_1d['ema_26']:
                strength_score += 20
                momentum_score += 25

            # Volume surge (15%)
            if 'Volume' in primary_data.columns and not primary_data['Volume'].empty:
                avg_volume = primary_data['Volume'].mean()
                current_volume = indicators_1d['volume']
                if current_volume > avg_volume * 1.5:
                    strength_score += 15
                    momentum_score += 20

            # Price action (10%)
            if len(primary_data) >= 2:
                price_change = (primary_data['Close'].iloc[-1] - primary_data['Close'].iloc[-2]) / primary_data['Close'].iloc[-2] * 100
                if price_change > 0:
                    strength_score += 10
                    momentum_score += 15
            else:
                price_change = 0.0

            # Multi-timeframe confirmation
            if not data_1h.empty and not data_15m.empty:
                indicators_1h = self.calculate_technical_indicators(data_1h)
                indicators_15m = self.calculate_technical_indicators(data_15m)

                if indicators_1h['rsi'] > 50 and indicators_15m['rsi'] > 50:
                    momentum_score += 10

            # Generate recommendation
            recommendation = self.generate_segment_recommendation(
                segment, strength_score, momentum_score, indicators_1d, price_change
            )

            # Save to database with error handling
            try:
                conn = self.db_manager.sqlite3.connect(self.db_manager.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO segment_analysis (
                        segment, strength_score, momentum_score, volatility,
                        volume, rsi, macd, ema_status, recommendation, reasoning
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    segment, strength_score, momentum_score, indicators_1d['volatility'],
                    indicators_1d['volume'], rsi, indicators_1d['macd'],
                    'BULLISH' if indicators_1d['ema_12'] > indicators_1d['ema_26'] else 'BEARISH',
                    recommendation['action'], recommendation['reasoning']
                ))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Database save error for {segment}: {e}")

            return {
                'segment': segment,
                'strength_score': strength_score,
                'momentum_score': momentum_score,
                'indicators': indicators_1d,
                'recommendation': recommendation,
                'price_change': price_change,
                'lot_size': self.segments[segment]['lot_size']
            }

        except Exception as e:
            print(f"Error analyzing {segment}: {e}")
            return None

    def generate_segment_recommendation(self, segment: str, strength: float, momentum: float, 
                                      indicators: Dict, price_change: float) -> Dict:
        """Generate AI-powered recommendation for segment"""

        try:
            # Determine action based on scores
            if strength >= 70 and momentum >= 60:
                if indicators['rsi'] < 70:  # Not overbought
                    action = "STRONG BUY"
                    option_type = "CE"
                    confidence = min(95, strength + momentum/2)
                else:
                    action = "CAUTIOUS BUY"
                    option_type = "CE"
                    confidence = 70
            elif strength >= 50 and momentum >= 40:
                action = "BUY"
                option_type = "CE"
                confidence = min(80, strength + momentum/3)
            elif strength <= 30 and momentum <= 30:
                action = "STRONG SELL"
                option_type = "PE"
                confidence = min(90, 100 - strength)
            elif strength <= 50 and momentum <= 40:
                action = "SELL"
                option_type = "PE"
                confidence = min(75, 90 - strength)
            else:
                action = "HOLD"
                option_type = "NEUTRAL"
                confidence = 50

            # Calculate suggested parameters
            current_price = indicators['current_price'] if indicators['current_price'] > 0 else 25000
            lot_size = self.segments[segment]['lot_size']

            # Strike selection logic
            if option_type == "CE":
                if action == "STRONG BUY":
                    strike_distance = 0  # ATM
                    target_multiplier = 2.5
                else:
                    strike_distance = 50  # OTM
                    target_multiplier = 2.0
            elif option_type == "PE":
                if action == "STRONG SELL":
                    strike_distance = 0  # ATM
                    target_multiplier = 2.5
                else:
                    strike_distance = -50  # OTM
                    target_multiplier = 2.0
            else:
                strike_distance = 0
                target_multiplier = 1.0

            # Generate reasoning
            reasoning_parts = []
            reasoning_parts.append(f"RSI: {indicators['rsi']:.1f}")
            reasoning_parts.append(f"MACD: {'Bullish' if indicators['macd'] > indicators['macd_signal'] else 'Bearish'}")
            reasoning_parts.append(f"EMA: {'Uptrend' if indicators['ema_12'] > indicators['ema_26'] else 'Downtrend'}")
            reasoning_parts.append(f"Volatility: {indicators['volatility']:.1f}%")
            reasoning_parts.append(f"Price Change: {price_change:+.2f}%")

            reasoning = f"{action} signal based on: " + ", ".join(reasoning_parts)

            return {
                'action': action,
                'option_type': option_type,
                'confidence': confidence,
                'reasoning': reasoning,
                'suggested_strike': current_price + strike_distance,
                'lot_size': lot_size,
                'target_multiplier': target_multiplier,
                'current_price': current_price
            }

        except Exception as e:
            print(f"Recommendation generation error: {e}")
            return {
                'action': 'HOLD',
                'option_type': 'NEUTRAL',
                'confidence': 50,
                'reasoning': 'Unable to generate recommendation due to data issues',
                'suggested_strike': 25000,
                'lot_size': 25,
                'target_multiplier': 1.0,
                'current_price': 25000
            }

    def get_global_sentiment(self) -> Dict:
        """Get global market sentiment"""
        try:
            global_sentiment = {
                'dow_change': 0.0,
                'nasdaq_change': 0.0,
                'sp500_change': 0.0,
                'overall_sentiment': 'NEUTRAL'
            }

            # Get US markets with error handling
            try:
                dow = yf.download("^DJI", period="2d", interval="1d", auto_adjust=True)
                if not dow.empty and len(dow) >= 2:
                    global_sentiment['dow_change'] = float((dow['Close'].iloc[-1] - dow['Close'].iloc[-2]) / dow['Close'].iloc[-2] * 100)
            except Exception as e:
                print(f"DOW data error: {e}")

            try:
                nasdaq = yf.download("^IXIC", period="2d", interval="1d", auto_adjust=True)
                if not nasdaq.empty and len(nasdaq) >= 2:
                    global_sentiment['nasdaq_change'] = float((nasdaq['Close'].iloc[-1] - nasdaq['Close'].iloc[-2]) / nasdaq['Close'].iloc[-2] * 100)
            except Exception as e:
                print(f"NASDAQ data error: {e}")

            try:
                sp500 = yf.download("^GSPC", period="2d", interval="1d", auto_adjust=True)
                if not sp500.empty and len(sp500) >= 2:
                    global_sentiment['sp500_change'] = float((sp500['Close'].iloc[-1] - sp500['Close'].iloc[-2]) / sp500['Close'].iloc[-2] * 100)
            except Exception as e:
                print(f"S&P500 data error: {e}")

            # Calculate overall sentiment
            changes = [global_sentiment['dow_change'], global_sentiment['nasdaq_change'], global_sentiment['sp500_change']]
            valid_changes = [x for x in changes if x != 0.0]

            if valid_changes:
                avg_change = sum(valid_changes) / len(valid_changes)

                if avg_change > 1:
                    global_sentiment['overall_sentiment'] = 'VERY POSITIVE'
                elif avg_change > 0.5:
                    global_sentiment['overall_sentiment'] = 'POSITIVE'
                elif avg_change > -0.5:
                    global_sentiment['overall_sentiment'] = 'NEUTRAL'
                elif avg_change > -1:
                    global_sentiment['overall_sentiment'] = 'NEGATIVE'
                else:
                    global_sentiment['overall_sentiment'] = 'VERY NEGATIVE'

            return global_sentiment

        except Exception as e:
            print(f"Error getting global sentiment: {e}")
            return {
                'dow_change': 0.0,
                'nasdaq_change': 0.0,
                'sp500_change': 0.0,
                'overall_sentiment': 'NEUTRAL'
            }

    def analyze_all_segments(self) -> List[Dict]:
        """Analyze all segments and rank them"""
        segment_analyses = []

        # Use segments from original mapping
        for segment in self.segments.keys():
            try:
                analysis = self.analyze_segment_strength(segment)
                if analysis:
                    segment_analyses.append(analysis)
                else:
                    print(f"Skipping {segment} due to analysis failure")
            except Exception as e:
                print(f"Error in segment analysis for {segment}: {e}")
                continue

        # Sort by combined score (strength + momentum)
        try:
            segment_analyses.sort(
                key=lambda x: x['strength_score'] + x['momentum_score'], 
                reverse=True
            )
        except Exception as e:
            print(f"Error sorting segments: {e}")

        return segment_analyses

    def generate_ai_market_message(self, analyses: List[Dict], global_sentiment: Dict) -> str:
        """Generate AI-powered market message"""
        try:
            if not analyses:
                return "Market analysis in progress... Stay tuned for signals! 🚀"

            # Prepare market data for AI
            market_summary = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'top_segment': analyses[0]['segment'] if analyses else 'None',
                'global_sentiment': global_sentiment['overall_sentiment'],
                'segment_data': analyses[:3]  # Top 3 segments
            }

            prompt = f"""
            आपको भारतीय शेयर बाजार का विशेषज्ञ ट्रेडिंग एडवाइजर बनकर एक संक्षिप्त और प्रभावी मैसेज लिखना है।

            वर्तमान मार्केट डेटा:
            समय: {market_summary['timestamp']}
            सबसे मजबूत सेगमेंट: {market_summary['top_segment']}
            ग्लोबल सेंटिमेंट: {market_summary['global_sentiment']}

            कृपया एक छोटा, स्पष्ट और actionable मैसेज लिखें (50 words में) जो:
            1. मुख्य recommendation दे
            2. Risk के बारे में बताए
            3. Motivational tone हो
            4. Hindi + English mix में हो

            उदाहरण: "Nifty में strong momentum दिख रहा है! RSI 65 पर ideal entry point. 25800 CE recommend, 2 lots safe. Global cues positive हैं. Stay disciplined! 💪"
            """

            result = self.gemini_api.generate_analysis(prompt)
            if result and result.get('success'):
                return result['response'].strip()
            else:
                return f"Market Update: {market_summary['top_segment']} showing strength. Global sentiment: {global_sentiment['overall_sentiment']}. Trade wisely! 📈"

        except Exception as e:
            print(f"AI message generation error: {e}")
            return f"Market analysis running... Top segment signals coming soon! 🚀"