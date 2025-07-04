
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional

class DatabaseManager:
    def __init__(self, db_path='trader_friend.db'):
        self.db_path = db_path
        self.init_advanced_database()
    
    def init_advanced_database(self):
        """Initialize advanced database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Signals log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                segment TEXT NOT NULL,
                instrument TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                entry_price REAL,
                target_price REAL,
                stop_loss REAL,
                lot_size INTEGER,
                premium REAL,
                reasoning TEXT,
                volatility REAL,
                rsi REAL,
                macd REAL,
                oi_buildup TEXT,
                pcr REAL,
                global_sentiment TEXT,
                news_sentiment TEXT,
                confidence_score REAL,
                status TEXT DEFAULT 'ACTIVE',
                exit_price REAL,
                exit_timestamp DATETIME,
                pnl REAL
            )
        ''')
        
        # Strategy configuration table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT UNIQUE NOT NULL,
                parameters TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Capital and risk management
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS capital_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_capital REAL,
                available_capital REAL,
                deployed_capital REAL,
                daily_pnl REAL,
                weekly_pnl REAL,
                monthly_pnl REAL,
                max_drawdown REAL,
                risk_percentage REAL
            )
        ''')
        
        # Backtest results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT,
                instrument TEXT,
                timeframe_start DATETIME,
                timeframe_end DATETIME,
                strategy_used TEXT,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                total_pnl REAL,
                win_rate REAL,
                max_profit REAL,
                max_loss REAL,
                sharpe_ratio REAL,
                trade_log TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Segment analysis
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS segment_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                segment TEXT NOT NULL,
                strength_score REAL,
                momentum_score REAL,
                volatility REAL,
                volume REAL,
                rsi REAL,
                macd REAL,
                ema_status TEXT,
                recommendation TEXT,
                reasoning TEXT
            )
        ''')
        
        # User interactions and emotions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_emotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                trade_id INTEGER,
                emotion_type TEXT,
                ai_message TEXT,
                user_response TEXT,
                effectiveness_score REAL
            )
        ''')
        
        # Market news and sentiment
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                news_headline TEXT,
                sentiment_score REAL,
                impact_level TEXT,
                source TEXT,
                category TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_signal(self, signal_data: Dict):
        """Log a trading signal"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signals_log (
                segment, instrument, signal_type, entry_price, target_price,
                stop_loss, lot_size, premium, reasoning, volatility, rsi,
                macd, oi_buildup, pcr, global_sentiment, news_sentiment,
                confidence_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal_data.get('segment'),
            signal_data.get('instrument'),
            signal_data.get('signal_type'),
            signal_data.get('entry_price'),
            signal_data.get('target_price'),
            signal_data.get('stop_loss'),
            signal_data.get('lot_size'),
            signal_data.get('premium'),
            signal_data.get('reasoning'),
            signal_data.get('volatility'),
            signal_data.get('rsi'),
            signal_data.get('macd'),
            signal_data.get('oi_buildup'),
            signal_data.get('pcr'),
            signal_data.get('global_sentiment'),
            signal_data.get('news_sentiment'),
            signal_data.get('confidence_score')
        ))
        
        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return signal_id
    
    def update_capital(self, capital_data: Dict):
        """Update capital information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO capital_logs (
                total_capital, available_capital, deployed_capital,
                daily_pnl, weekly_pnl, monthly_pnl, max_drawdown, risk_percentage
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            capital_data.get('total_capital'),
            capital_data.get('available_capital'),
            capital_data.get('deployed_capital'),
            capital_data.get('daily_pnl'),
            capital_data.get('weekly_pnl'),
            capital_data.get('monthly_pnl'),
            capital_data.get('max_drawdown'),
            capital_data.get('risk_percentage')
        ))
        
        conn.commit()
        conn.close()
    
    def save_backtest_result(self, backtest_data: Dict):
        """Save backtest results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO backtest_results (
                test_name, instrument, timeframe_start, timeframe_end,
                strategy_used, total_trades, winning_trades, losing_trades,
                total_pnl, win_rate, max_profit, max_loss, sharpe_ratio, trade_log
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            backtest_data.get('test_name'),
            backtest_data.get('instrument'),
            backtest_data.get('timeframe_start'),
            backtest_data.get('timeframe_end'),
            backtest_data.get('strategy_used'),
            backtest_data.get('total_trades'),
            backtest_data.get('winning_trades'),
            backtest_data.get('losing_trades'),
            backtest_data.get('total_pnl'),
            backtest_data.get('win_rate'),
            backtest_data.get('max_profit'),
            backtest_data.get('max_loss'),
            backtest_data.get('sharpe_ratio'),
            json.dumps(backtest_data.get('trade_log', []))
        ))
        
        conn.commit()
        conn.close()
    
    def get_active_signals(self):
        """Get all active signals"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM signals_log WHERE status = 'ACTIVE' ORDER BY timestamp DESC",
            conn
        )
        conn.close()
        return df
    
    def get_performance_summary(self, days=30):
        """Get performance summary for last N days"""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent trades
        trades_df = pd.read_sql_query(f'''
            SELECT * FROM signals_log 
            WHERE timestamp >= datetime('now', '-{days} days')
            AND exit_price IS NOT NULL
            ORDER BY timestamp DESC
        ''', conn)
        
        # Get capital history
        capital_df = pd.read_sql_query(f'''
            SELECT * FROM capital_logs 
            WHERE timestamp >= datetime('now', '-{days} days')
            ORDER BY timestamp DESC
        ''', conn)
        
        conn.close()
        
        return {
            'trades': trades_df,
            'capital_history': capital_df,
            'total_trades': len(trades_df),
            'winning_trades': len(trades_df[trades_df['pnl'] > 0]),
            'losing_trades': len(trades_df[trades_df['pnl'] < 0]),
            'total_pnl': trades_df['pnl'].sum() if not trades_df.empty else 0,
            'win_rate': len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100 if not trades_df.empty else 0
        }
