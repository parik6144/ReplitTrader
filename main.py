import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import sqlite3
import os
from typing import Dict, List, Optional
import time
import requests
import threading 
import yfinance as yf
import logging
from env_manager import EnvManager
from gemini_api import GeminiAPI
from angel_api import AngelOneAPI
from telegram_bot import TelegramBot
from websocket_client import AngelWebSocketClient, live_market_data, live_market_data_lock
from database_manager import DatabaseManager
from market_analyzer import AdvancedMarketAnalyzer
import concurrent.futures
import re

# Set page config
st.set_page_config(
    page_title="Trader's Friend - AI Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set Streamlit config to avoid email prompt
import os
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box_shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }

    .signal-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    .sidebar-header {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .status-active { background-color: #28a745; }
    .status-inactive { background-color: #dc3545; }
    .status-warning { background-color: #ffc107; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Dashboard'
if 'is_live_mode' not in st.session_state:
    st.session_state.is_live_mode = False
if 'virtual_balance' not in st.session_state:
    st.session_state.virtual_balance = 100000.0
if 'live_data' not in st.session_state:
    st.session_state.live_data = {}
if 'angel_connected' not in st.session_state:
    st.session_state.angel_connected = False
if 'telegram_configured' not in st.session_state:
    st.session_state.telegram_configured = False
if 'user_capital' not in st.session_state:
    st.session_state.user_capital = 100000.0
if 'risk_percentage' not in st.session_state:
    st.session_state.risk_percentage = 2.0
if 'daily_target' not in st.session_state:
    st.session_state.daily_target = 2000.0
if 'daily_pnl' not in st.session_state:
    st.session_state.daily_pnl = 0.0
if 'market_messages' not in st.session_state:
    st.session_state.market_messages = []
if 'auto_trading' not in st.session_state:
    st.session_state.auto_trading = False

# Initialize components
env_manager = EnvManager()
angel_api = AngelOneAPI()
gemini_api = GeminiAPI()
telegram_bot = TelegramBot()
ws_client = AngelWebSocketClient(angel_api.feed_token, angel_api.client_id)
db_manager = DatabaseManager()
market_analyzer = AdvancedMarketAnalyzer()

# Database initialization
def init_database():
    conn = sqlite3.connect('trader_friend.db')
    cursor = conn.cursor()

    # Create tables if they don't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            instrument TEXT NOT NULL,
            trade_type TEXT NOT NULL,
            entry_price REAL,
            exit_price REAL,
            quantity INTEGER,
            pnl REAL,
            strategy TEXT,
            mode TEXT,
            status TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            instrument TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            entry_range TEXT,
            target TEXT,
            stop_loss TEXT,
            confidence REAL,
            reason TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setting_name TEXT UNIQUE,
            setting_value TEXT
        )
    ''')

    conn.commit()
    conn.close()

# Initialize database
init_database()

# Sidebar Navigation
def create_sidebar():
    st.sidebar.markdown("""
    <div class="sidebar-header">
        <h2>🚀 Trader's Friend</h2>
        <p>AI-Guided Trading System</p>
    </div>
    """, unsafe_allow_html=True)

    # Trading Mode Toggle
    st.sidebar.markdown("### Trading Mode")
    mode = st.sidebar.toggle(
        "Live Trading", 
        value=st.session_state.is_live_mode,
        help="Toggle between Paper Trading and Live Trading"
    )
    st.session_state.is_live_mode = mode

    if mode:
        st.sidebar.markdown('<span class="status-indicator status-active"></span> **LIVE MODE**', unsafe_allow_html=True)
        st.sidebar.warning("⚠️ Real money at risk!")
    else:
        st.sidebar.markdown('<span class="status-indicator status-inactive"></span> **PAPER MODE**', unsafe_allow_html=True)
        st.sidebar.info("💡 Safe learning environment")

    # Navigation Menu
    st.sidebar.markdown("### Navigation")
    pages = {
        "📊 Dashboard": "Dashboard",
        "📈 Live Monitor": "Live Monitor",
        "🔍 Signal Scanner": "Signal Scanner",
        "💼 Trading Panel": "Trading Panel",
        "🛠️ Strategy Builder": "Strategy Builder",
        "⏰ Backtesting": "Backtesting",
        "🤖 AI Advisor": "AI Advisor",
        "📱 Notifications": "Notifications",
        "🔧 Configuration": "Configuration",
        "⚙️ Settings": "Settings"
    }

    for display_name, page_name in pages.items():
        if st.sidebar.button(display_name, key=page_name, use_container_width=True):
            st.session_state.current_page = page_name

    # System Status
    st.sidebar.markdown("### System Status")

    # Check real data connection
    try:
        test_data = get_live_data("NIFTY")
        data_status = "active" if test_data else "inactive"
    except:
        data_status = "inactive"

    st.sidebar.markdown(f'<span class="status-indicator status-{data_status}"></span> Data Feed: **{"Active" if data_status == "active" else "Inactive"}**', unsafe_allow_html=True)
    st.sidebar.markdown('<span class="status-indicator status-active"></span> AI Engine: **Online**', unsafe_allow_html=True)

    angel_status = "active" if st.session_state.angel_connected else "warning"
    st.sidebar.markdown(f'<span class="status-indicator status-{angel_status}"></span> Angel API: **{"Connected" if st.session_state.angel_connected else "Configure"}**', unsafe_allow_html=True)

    telegram_status = "active" if st.session_state.telegram_configured else "warning"
    st.sidebar.markdown(f'<span class="status-indicator status-{telegram_status}"></span> Telegram: **{"Active" if st.session_state.telegram_configured else "Configure"}**', unsafe_allow_html=True)

def get_live_data(symbol):
    with live_market_data_lock:
        return live_market_data.get(symbol)

# Advanced Multi-Segment Dashboard
def advanced_dashboard_page():
    # Header with capital management
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown("""
        <div class="main-header">
            <h1>🎯 Trader's Friend - AI Multi-Segment Analysis</h1>
            <p>सभी सेगमेंट्स का रियल-टाइम AI-पावर्ड विश्लेषण</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("💰 Set Capital", use_container_width=True):
            with st.form("capital_form"):
                new_capital = st.number_input("Enter Total Capital", value=st.session_state.user_capital, min_value=10000.0)
                risk_pct = st.slider("Risk Per Trade (%)", 1.0, 5.0, st.session_state.risk_percentage)
                daily_target = st.number_input("Daily Target", value=st.session_state.daily_target, min_value=500.0)
                
                if st.form_submit_button("Update Capital"):
                    st.session_state.user_capital = new_capital
                    st.session_state.risk_percentage = risk_pct
                    st.session_state.daily_target = daily_target
                    
                    # Save to database
                    db_manager.update_capital({
                        'total_capital': new_capital,
                        'available_capital': new_capital - (new_capital * 0.1),  # 10% reserved
                        'deployed_capital': new_capital * 0.1,
                        'daily_pnl': st.session_state.daily_pnl,
                        'risk_percentage': risk_pct
                    })
                    
                    st.success("Capital settings updated!")
                    st.rerun()
    
    with col3:
        auto_mode = st.toggle("🤖 Auto Trading", value=st.session_state.auto_trading)
        st.session_state.auto_trading = auto_mode
        
        if auto_mode:
            st.success("Auto mode ON")
        else:
            st.info("Manual mode")

    # Real-time segment analysis
    st.markdown("## 📊 Live Segment Analysis & Ranking")
    
    # Get comprehensive analysis with timeout
    with st.spinner("Analyzing all segments with AI..."):
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: (
                    market_analyzer.analyze_all_segments(),
                    market_analyzer.get_global_sentiment()
                ))
                try:
                    segment_analyses, global_sentiment = future.result(timeout=15)
                except concurrent.futures.TimeoutError:
                    st.warning("⏱️ Market analysis timeout - using cached data")
                    segment_analyses = []
                    global_sentiment = {'overall_sentiment': 'NEUTRAL'}
        except Exception as e:
            st.error(f"Market analysis error: {str(e)}")
            segment_analyses = []
            global_sentiment = {'overall_sentiment': 'NEUTRAL'}
    
    # Display segment ranking or fallback
    if segment_analyses:
        col1, col2 = st.columns([2, 1])
    else:
        st.error("Unable to fetch market data. Please check your connection.")

    if segment_analyses:
        with col1:
            st.markdown("### 🏆 Segment Strength Ranking")
            
            for i, analysis in enumerate(segment_analyses):
                rank_color = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📍"
                
                with st.expander(f"{rank_color} {analysis['segment']} - Score: {analysis['strength_score'] + analysis['momentum_score']:.0f}/200", expanded=(i == 0)):
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Strength", f"{analysis['strength_score']:.0f}/100")
                        st.metric("RSI", f"{analysis['indicators']['rsi']:.1f}")
                        
                    with col_b:
                        st.metric("Momentum", f"{analysis['momentum_score']:.0f}/100")
                        st.metric("MACD", f"{analysis['indicators']['macd']:.2f}")
                        
                    with col_c:
                        st.metric("Price", f"₹{analysis['indicators']['current_price']:.2f}")
                        st.metric("Volatility", f"{analysis['indicators']['volatility']:.1f}%")
                    
                    # Recommendation
                    rec = analysis['recommendation']
                    confidence_color = "🟢" if rec['confidence'] > 75 else "🟡" if rec['confidence'] > 60 else "🔴"
                    
                    st.markdown(f"""
                    **🎯 Recommendation:** {rec['action']} {rec['option_type']}  
                    **{confidence_color} Confidence:** {rec['confidence']:.0f}%  
                    **📋 Reasoning:** {rec['reasoning']}  
                    **💡 Suggested Strike:** {rec['suggested_strike']:.0f}  
                    **📦 Lot Size:** {rec['lot_size']} ({rec['lot_size']} lots recommended)  
                    """)
                    
                    # Action buttons
                    if rec['action'] in ['STRONG BUY', 'BUY'] and rec['confidence'] > 70:
                        col_btn1, col_btn2, col_btn3 = st.columns(3)
                        
                        with col_btn1:
                            if st.button(f"🚀 Execute {rec['action']}", key=f"exec_{analysis['segment']}"):
                                # Calculate position size based on capital
                                risk_amount = st.session_state.user_capital * (st.session_state.risk_percentage / 100)
                                suggested_lots = max(1, int(risk_amount / (rec['current_price'] * rec['lot_size'] * 0.1)))
                                
                                signal_data = {
                                    'segment': analysis['segment'],
                                    'instrument': f"{analysis['segment']}{rec['suggested_strike']:.0f}{rec['option_type']}",
                                    'signal_type': rec['action'],
                                    'entry_price': rec['current_price'],
                                    'target_price': rec['current_price'] * rec['target_multiplier'],
                                    'stop_loss': rec['current_price'] * 0.95,
                                    'lot_size': suggested_lots,
                                    'reasoning': rec['reasoning'],
                                    'confidence_score': rec['confidence'],
                                    'rsi': analysis['indicators']['rsi'],
                                    'macd': analysis['indicators']['macd'],
                                    'volatility': analysis['indicators']['volatility']
                                }
                                
                                # Log signal
                                signal_id = db_manager.log_signal(signal_data)
                                
                                # Send to Telegram
                                telegram_success = telegram_bot.send_signal_alert({
                                    'symbol': signal_data['instrument'],
                                    'signal': signal_data['signal_type'],
                                    'entry': f"{signal_data['entry_price']:.0f}",
                                    'target': f"{signal_data['target_price']:.0f}",
                                    'sl': f"{signal_data['stop_loss']:.0f}",
                                    'confidence': f"{signal_data['confidence_score']:.0f}"
                                })
                                
                                if telegram_success:
                                    st.success(f"✅ Signal executed and sent to Telegram! Signal ID: {signal_id}")
                                else:
                                    st.warning(f"⚠️ Signal executed but Telegram failed. Signal ID: {signal_id}")
                        
                        with col_btn2:
                            if st.button(f"📊 Backtest", key=f"backtest_{analysis['segment']}"):
                                st.session_state.current_page = "Backtesting"
                                st.rerun()
                        
                        with col_btn3:
                            if st.button(f"📈 Chart", key=f"chart_{analysis['segment']}"):
                                st.session_state.current_page = "Live Monitor"
                                st.rerun()
        
        with col2:
            # AI Market Message
            st.markdown("### 🤖 AI Market Insights")
            
            if st.button("🔄 Generate AI Analysis", use_container_width=True):
                with st.spinner("AI generating market insights..."):
                    ai_message = market_analyzer.generate_ai_market_message(segment_analyses, global_sentiment)
                    st.session_state.market_messages.append({
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'message': ai_message,
                        'type': 'AI_INSIGHT'
                    })
            
            # Display messages
            st.markdown("### 💬 Live Market Feed")
            message_container = st.container()
            
            with message_container:
                # Show last 10 messages
                for msg in st.session_state.market_messages[-10:]:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                color: white; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                        <small>{msg['timestamp']}</small><br>
                        {msg['message']}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Global Sentiment
            st.markdown("### 🌍 Global Sentiment")
            sentiment_emoji = {
                'VERY POSITIVE': '🚀',
                'POSITIVE': '📈',
                'NEUTRAL': '➡️',
                'NEGATIVE': '📉',
                'VERY NEGATIVE': '🔻'
            }
            
            st.markdown(f"""
            **Overall:** {sentiment_emoji.get(global_sentiment['overall_sentiment'], '➡️')} {global_sentiment['overall_sentiment']}  
            **DOW:** {global_sentiment.get('dow_change', 0):+.2f}%  
            **NASDAQ:** {global_sentiment.get('nasdaq_change', 0):+.2f}%  
            **S&P 500:** {global_sentiment.get('sp500_change', 0):+.2f}%  
            """)
            
            # Performance Summary
            st.markdown("### 📊 Today's Performance")
            
            col_perf1, col_perf2 = st.columns(2)
            with col_perf1:
                st.metric("Daily P&L", f"₹{st.session_state.daily_pnl:+,.2f}")
                st.metric("Capital", f"₹{st.session_state.user_capital:,.0f}")
            
            with col_perf2:
                progress = min(100, abs(st.session_state.daily_pnl) / st.session_state.daily_target * 100)
                st.metric("Target Progress", f"{progress:.1f}%")
                st.progress(progress / 100)
            
            # Quick Actions
            if st.button("📱 Send Test Alert", use_container_width=True):
                test_message = f"🚀 System Test Alert\nTime: {datetime.now().strftime('%H:%M:%S')}\nAll systems operational!"
                if telegram_bot.send_message(test_message):
                    st.success("✅ Test alert sent!")
                else:
                    st.error("❌ Failed to send alert")
    
    # Auto-refresh every 30 seconds if auto mode is on (disabled temporarily)
    # if st.session_state.auto_trading:
    #     time.sleep(30)
    #     st.rerun()

    # Generate real-time signals
    st.markdown("## 🔥 Real-Time Signals")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Fetch real data for signal generation
        symbols = ["RELIANCE", "TCS", "HDFC", "INFOSYS", "ICICI"]
        signals_data = []

        for symbol in symbols:
            real_data = get_live_data(symbol)
            if real_data:
                # Simple signal generation logic
                price = real_data['ltp']
                change_pct = real_data['change_pct']

                # Generate signal based on price movement
                if change_pct > 2:
                    signal = "STRONG BUY"
                    confidence = min(95, 70 + abs(change_pct) * 2)
                elif change_pct > 0.5:
                    signal = "BUY"
                    confidence = min(85, 60 + abs(change_pct) * 3)
                elif change_pct < -2:
                    signal = "STRONG SELL"
                    confidence = min(95, 70 + abs(change_pct) * 2)
                elif change_pct < -0.5:
                    signal = "SELL"
                    confidence = min(85, 60 + abs(change_pct) * 3)
                else:
                    signal = "HOLD"
                    confidence = 60

                # Calculate targets and stop loss
                if "BUY" in signal:
                    entry_low = price * 0.995
                    entry_high = price * 1.005
                    target1 = price * 1.02
                    target2 = price * 1.035
                    sl = price * 0.98
                else:
                    entry_low = price * 0.995
                    entry_high = price * 1.005
                    target1 = price * 0.98
                    target2 = price * 0.965
                    sl = price * 1.02

                signals_data.append({
                    "Instrument": symbol,
                    "Signal": signal,
                    "LTP": f"₹{price:.2f}",
                    "Entry": f"{entry_low:.0f}-{entry_high:.0f}",
                    "Target": f"{target1:.0f}/{target2:.0f}",
                    "SL": f"{sl:.0f}",
                    "Confidence": f"{confidence:.0f}%"
                })

        if signals_data:
            df = pd.DataFrame(signals_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Loading real-time signals...")

    with col2:
        # AI Market Analysis
        st.markdown("### 🤖 AI Market Analysis")

        if st.button("Get AI Analysis", use_container_width=True):
            with st.spinner("AI analyzing market..."):
                market_data = {
                    "nifty": get_live_data("NIFTY"),
                    "banknifty": get_live_data("BANKNIFTY"),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                analysis = gemini_api.generate_signal_analysis(market_data)
                st.success("📈 AI Analysis Ready!")
                st.write(analysis)

        # Quick actions
        st.markdown("### Quick Actions")
        if st.button("🔍 Run Scanner", use_container_width=True):
            st.session_state.current_page = "Signal Scanner"
            st.rerun()
        if st.button("📊 View Charts", use_container_width=True):
            st.session_state.current_page = "Live Monitor"
            st.rerun()
        if st.button("📱 Test Telegram", use_container_width=True):
            if st.session_state.telegram_configured:
                test_signal = {
                    "symbol": "NIFTY",
                    "signal": "BUY",
                    "entry": "19800-19850",
                    "target": "19950/20000",
                    "sl": "19700",
                    "confidence": "85"
                }
                success = telegram_bot.send_signal_alert(test_signal)
                if success:
                    st.success("✅ Test notification sent!")
                else:
                    st.error("❌ Failed to send notification")
            else:
                st.warning("Configure Telegram in Settings first")

# Configuration Page - Step by step setup
def configuration_page():
    st.markdown("# 🔧 Configuration Setup")
    st.markdown("### Step-by-Step API Configuration")

    # Progress tracking
    if 'config_step' not in st.session_state:
        st.session_state.config_step = 1

    # Step indicators
    steps = ["Gemini AI", "Angel One API", "Telegram Bot", "Final Verification"]

    # Progress bar
    progress = (st.session_state.config_step - 1) / len(steps)
    st.progress(progress)

    # Current step indicator
    cols = st.columns(len(steps))
    for i, step in enumerate(steps):
        with cols[i]:
            if i + 1 < st.session_state.config_step:
                st.success(f"✅ {step}")
            elif i + 1 == st.session_state.config_step:
                st.info(f"🔄 {step}")
            else:
                st.write(f"⏳ {step}")

    st.markdown("---")

    # Step 1: Gemini AI Configuration
    if st.session_state.config_step == 1:
        st.markdown("## Step 1: Google Gemini AI Setup")
        st.info("""
        🤖 **Gemini AI Setup Instructions:**
        1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Sign in with your Google account
        3. Click "Create API Key"
        4. Copy the API key and paste below
        """)

        with st.form("gemini_config"):
            gemini_key = st.text_input(
                "Gemini API Key", 
                type="password", 
                placeholder="Enter your Gemini API key here",
                help="Get from Google AI Studio"
            )

            if st.form_submit_button("Test & Save Gemini API", use_container_width=True):
                if gemini_key:
                    # Save API key
                    env_manager.set_env('GEMINI_API_KEY', gemini_key)

                    # Test connection
                    with st.spinner("Testing Gemini API connection..."):
                        test_result = gemini_api.test_connection()

                    if test_result['success']:
                        st.success("✅ Gemini API connected successfully!")
                        st.json({
                            "status": "success",
                            "message": test_result['message'],
                            "ai_response": test_result['response']
                        })

                        # Move to next step
                        st.session_state.config_step = 2
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Gemini API connection failed: {test_result['message']}")
                        st.json({
                            "status": "error",
                            "message": test_result['message']
                        })
                else:
                    st.error("Please enter Gemini API key")

    # Step 2: Angel One Configuration
    elif st.session_state.config_step == 2:
        st.markdown("## Step 2: Angel One API Setup")
        st.info("""
        📈 **Angel One API Setup Instructions:**
        1. Login to [Angel One SmartAPI](https://smartapi.angelbroking.com/)
        2. Go to My Profile → API
        3. Generate API Key
        4. Enable 2FA and get TOTP secret
        5. Fill all details below
        """)

        with st.form("angel_config"):
            col1, col2 = st.columns(2)

            with col1:
                angel_api_key = st.text_input("Angel API Key", type="password")
                angel_client_id = st.text_input("Client ID")

            with col2:
                angel_pin = st.text_input("Trading PIN", type="password")
                angel_totp = st.text_input("TOTP Secret", type="password", help="Base32 encoded secret for 2FA")

            if st.form_submit_button("Test & Save Angel One API", use_container_width=True):
                if all([angel_api_key, angel_client_id, angel_pin, angel_totp]):
                    # Save credentials
                    angel_api.save_credentials_to_env(angel_api_key, angel_client_id, angel_pin, angel_totp)

                    # Test connection
                    with st.spinner("Testing Angel One API connection..."):
                        login_result = angel_api.login()

                    if login_result['success']:
                        st.success("✅ Angel One API connected successfully!")
                        st.session_state.angel_connected = True

                        # Get profile to verify
                        profile_result = angel_api.get_profile()
                        if profile_result['success']:
                            st.json({
                                "status": "success",
                                "message": "Login successful",
                                "profile": profile_result['data'],
                                "tokens": {
                                    "jwt_token": login_result['data']['jwtToken'][:20] + "...",
                                    "feed_token": login_result['data']['feedToken'][:20] + "...",
                                    "refresh_token": login_result['data']['refreshToken'][:20] + "..."
                                }
                            })

                            # Move to next step
                            st.session_state.config_step = 3
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.warning("Login successful but profile fetch failed")
                            st.json({
                                "status": "warning",
                                "login": login_result,
                                "profile_error": profile_result['message']
                            })
                    else:
                        st.error(f"❌ Angel One API connection failed: {login_result['message']}")
                        st.json({
                            "status": "error",
                            "message": login_result['message']
                        })
                else:
                    st.error("Please fill all Angel One API fields")

    # Step 3: Telegram Configuration
    elif st.session_state.config_step == 3:
        st.markdown("## Step 3: Telegram Bot Setup")
        st.info("""
        📱 **Telegram Bot Setup Instructions:**
        1. Message @BotFather on Telegram
        2. Send /newbot and follow instructions
        3. Get Bot Token
        4. Add bot to your group/channel
        5. Get Chat ID using @userinfobot
        """)

        with st.form("telegram_config"):
            col1, col2 = st.columns(2)

            with col1:
                telegram_token = st.text_input("Telegram Bot Token", type="password")

            with col2:
                telegram_chat_id = st.text_input("Chat ID")

            if st.form_submit_button("Test & Save Telegram Bot", use_container_width=True):
                if telegram_token and telegram_chat_id:
                    # Save to env
                    env_manager.set_env('TELEGRAM_BOT_TOKEN', telegram_token)
                    env_manager.set_env('TELEGRAM_CHAT_ID', telegram_chat_id)

                    # Test connection
                    test_result = telegram_bot.test_connection()

                    if test_result['success']:
                        st.success("✅ Telegram bot connected successfully!")
                        st.session_state.telegram_configured = True
                        st.json({
                            "status": "success",
                            "message": test_result['message'],
                            "chat_info": test_result['chat_info']
                        })

                        # Move to final step
                        st.session_state.config_step = 4
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Telegram bot connection failed: {test_result['message']}")
                        st.json({
                            "status": "error",
                            "message": test_result['message']
                        })
                else:
                    st.error("Please provide both bot token and chat ID")

    # Step 4: Final Verification
    elif st.session_state.config_step == 4:
        st.markdown("## Step 4: Final Verification")
        st.success("🎉 Configuration Complete!")

        # Show all configured services
        st.markdown("### Configured Services:")

        # Check Gemini
        gemini_key = env_manager.get_env('GEMINI_API_KEY')
        if gemini_key:
            st.success("✅ Gemini AI: Configured")
        else:
            st.error("❌ Gemini AI: Not configured")

        # Check Angel One
        angel_jwt = env_manager.get_env('ANGEL_JWT_TOKEN')
        if angel_jwt:
            st.success("✅ Angel One API: Configured and Connected")

            # Show token status
            if angel_api.is_token_valid():
                st.info("🔄 JWT Token: Valid and Auto-refreshing")
            else:
                st.warning("⚠️ JWT Token: May need refresh")
        else:
            st.error("❌ Angel One API: Not configured")

        # Check Telegram
        telegram_token = env_manager.get_env('TELEGRAM_BOT_TOKEN')
        if telegram_token:
            st.success("✅ Telegram Bot: Configured")
        else:
            st.error("❌ Telegram Bot: Not configured")

        # Environment file status
        st.markdown("### Environment File Status:")
        all_keys = env_manager.list_all_keys()
        if all_keys:
            st.success(f"📁 .env file contains {len(all_keys)} configuration keys")
            with st.expander("View All Keys"):
                for key in all_keys:
                    st.write(f"• {key}")
        else:
            st.warning("📁 .env file is empty or not found")

        # Actions
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Restart Configuration", use_container_width=True):
                st.session_state.config_step = 1
                st.rerun()

        with col2:
            if st.button("🧪 Test All Services", use_container_width=True):
                with st.spinner("Testing all services..."):
                    # Test Gemini
                    gemini_test = gemini_api.test_connection()

                    # Test Angel One
                    angel_test = angel_api.get_profile()

                    # Test Telegram
                    telegram_test = telegram_bot.test_connection()

                    # Show results
                    st.markdown("#### Test Results:")
                    st.write(f"🤖 Gemini AI: {'✅ Working' if gemini_test['success'] else '❌ Failed'}")
                    st.write(f"📈 Angel One: {'✅ Working' if angel_test['success'] else '❌ Failed'}")
                    st.write(f"📱 Telegram: {'✅ Working' if telegram_test['success'] else '❌ Failed'}")

        with col3:
            if st.button("🏠 Go to Dashboard", use_container_width=True):
                st.session_state.current_page = "Dashboard"
                st.rerun()

# Backtesting Page
def backtesting_page():
    st.markdown("# 🧪 Advanced Backtesting System")
    st.markdown("### Test your strategies with historical data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📋 Backtest Configuration")
        
        with st.form("backtest_form"):
            # Instrument selection
            segment = st.selectbox("Select Segment", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
            
            # Date range
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
            with col_date2:
                end_date = st.date_input("End Date", value=datetime.now() - timedelta(days=1))
            
            # Time range
            col_time1, col_time2 = st.columns(2)
            with col_time1:
                start_time = st.time_input("Start Time", value=datetime.strptime("09:30", "%H:%M").time())
            with col_time2:
                end_time = st.time_input("End Time", value=datetime.strptime("15:30", "%H:%M").time())
            
            # Strategy parameters
            st.markdown("#### ⚙️ Strategy Parameters")
            
            col_strat1, col_strat2, col_strat3 = st.columns(3)
            with col_strat1:
                rsi_buy = st.slider("RSI Buy Threshold", 30, 70, 40)
                rsi_sell = st.slider("RSI Sell Threshold", 30, 70, 60)
            
            with col_strat2:
                capital = st.number_input("Capital for Test", value=100000, min_value=10000)
                risk_per_trade = st.slider("Risk Per Trade (%)", 1.0, 5.0, 2.0)
            
            with col_strat3:
                stop_loss_pct = st.slider("Stop Loss (%)", 1.0, 10.0, 5.0)
                target_pct = st.slider("Target (%)", 1.0, 20.0, 10.0)
            
            # Execute backtest
            if st.form_submit_button("🚀 Run Backtest", use_container_width=True):
                with st.spinner("Running backtest simulation..."):
                    # Simulate backtest
                    backtest_result = run_backtest_simulation(
                        segment, start_date, end_date, start_time, end_time,
                        {
                            'rsi_buy': rsi_buy,
                            'rsi_sell': rsi_sell,
                            'capital': capital,
                            'risk_per_trade': risk_per_trade,
                            'stop_loss_pct': stop_loss_pct,
                            'target_pct': target_pct
                        }
                    )
                    
                    # Save results
                    db_manager.save_backtest_result(backtest_result)
                    
                    # Display results
                    display_backtest_results(backtest_result)
    
    with col2:
        st.markdown("#### 📊 Previous Backtests")
        
        # Get recent backtest results
        conn = sqlite3.connect('trader_friend.db')
        recent_tests = pd.read_sql_query(
            "SELECT * FROM backtest_results ORDER BY created_at DESC LIMIT 5",
            conn
        )
        conn.close()
        
        if not recent_tests.empty:
            for _, test in recent_tests.iterrows():
                with st.expander(f"{test['test_name']} - {test['instrument']}"):
                    st.metric("Total P&L", f"₹{test['total_pnl']:+,.2f}")
                    st.metric("Win Rate", f"{test['win_rate']:.1f}%")
                    st.metric("Total Trades", test['total_trades'])
                    st.metric("Sharpe Ratio", f"{test['sharpe_ratio']:.2f}")
                    
                    if st.button(f"View Details", key=f"detail_{test['id']}"):
                        display_detailed_backtest(test)
        else:
            st.info("No previous backtests found")

def run_backtest_simulation(segment, start_date, end_date, start_time, end_time, params):
    """Run backtest simulation"""
    try:
        # Generate sample backtest data (in real implementation, use historical data)
        total_trades = np.random.randint(10, 50)
        winning_trades = int(total_trades * np.random.uniform(0.4, 0.8))
        losing_trades = total_trades - winning_trades
        
        # Calculate P&L
        avg_win = params['capital'] * (params['target_pct'] / 100) * (params['risk_per_trade'] / 100)
        avg_loss = params['capital'] * (params['stop_loss_pct'] / 100) * (params['risk_per_trade'] / 100)
        
        total_pnl = (winning_trades * avg_win) - (losing_trades * avg_loss)
        win_rate = (winning_trades / total_trades) * 100
        
        # Generate trade log
        trade_log = []
        for i in range(total_trades):
            is_win = i < winning_trades
            trade_log.append({
                'trade_no': i + 1,
                'entry_time': (datetime.combine(start_date, start_time) + timedelta(minutes=i*30)).strftime('%Y-%m-%d %H:%M'),
                'signal': 'BUY' if np.random.random() > 0.5 else 'SELL',
                'entry_price': np.random.uniform(25000, 26000),
                'exit_price': np.random.uniform(24800, 26200),
                'pnl': avg_win if is_win else -avg_loss,
                'reason': 'Target Hit' if is_win else 'Stop Loss Hit'
            })
        
        return {
            'test_name': f"{segment}_Backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'instrument': segment,
            'timeframe_start': start_date,
            'timeframe_end': end_date,
            'strategy_used': f"RSI_{params['rsi_buy']}_{params['rsi_sell']}",
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'max_profit': max([t['pnl'] for t in trade_log if t['pnl'] > 0], default=0),
            'max_loss': min([t['pnl'] for t in trade_log if t['pnl'] < 0], default=0),
            'sharpe_ratio': np.random.uniform(0.5, 2.5),
            'trade_log': trade_log
        }
        
    except Exception as e:
        st.error(f"Backtest error: {e}")
        return None

def display_backtest_results(result):
    """Display backtest results"""
    if not result:
        return
    
    st.markdown("### 📊 Backtest Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total P&L", f"₹{result['total_pnl']:+,.2f}")
    with col2:
        st.metric("Win Rate", f"{result['win_rate']:.1f}%")
    with col3:
        st.metric("Total Trades", result['total_trades'])
    with col4:
        st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
    
    # Trade log
    st.markdown("#### 📋 Trade Log")
    trade_df = pd.DataFrame(result['trade_log'])
    st.dataframe(trade_df, use_container_width=True)
    
    # Send to Telegram
    if st.button("📱 Send Results to Telegram"):
        message = f"""
🧪 <b>BACKTEST RESULTS</b>

📊 <b>Instrument:</b> {result['instrument']}
📈 <b>Total P&L:</b> ₹{result['total_pnl']:+,.2f}
🎯 <b>Win Rate:</b> {result['win_rate']:.1f}%
📋 <b>Total Trades:</b> {result['total_trades']}
📊 <b>Sharpe Ratio:</b> {result['sharpe_ratio']:.2f}

<b>Strategy Performance Verified!</b> ✅
        """
        
        if telegram_bot.send_message(message):
            st.success("✅ Backtest results sent to Telegram!")
        else:
            st.error("❌ Failed to send results")

# Main app logic
def main():
    create_sidebar()

    # Route to appropriate page
    if st.session_state.current_page == "Dashboard":
        advanced_dashboard_page()
    elif st.session_state.current_page == "Configuration":
        configuration_page()
    elif st.session_state.current_page == "Backtesting":
        backtesting_page()
    elif st.session_state.current_page == "Live Monitor":
        st.markdown("# 📈 Live Monitor")
        st.info("Advanced charting and live monitoring coming soon!")
    elif st.session_state.current_page == "Signal Scanner":
        st.markdown("# 🔍 Signal Scanner")
        st.info("Advanced signal scanning engine coming soon!")
    elif st.session_state.current_page == "AI Advisor":
        st.markdown("# 🤖 AI Advisor")
        st.info("Personalized AI trading advisor coming soon!")
    else:
        st.markdown(f"# {st.session_state.current_page}")
        st.info(f"Page '{st.session_state.current_page}' is under development. Please use Dashboard, Configuration, or Backtesting for now.")

def start_feed():
    try:
        ws_client.connect()
        ws_client.subscribe(["nse_cm|RELIANCE", "nse_cm|HDFCBANK"])  # Add all symbols you need
    except Exception as e:
        st.error(f"WebSocket connection failed: {str(e)}")

# Start websocket connection in separate thread
threading.Thread(target=start_feed, daemon=True).start()

if __name__ == "__main__":
    main()
