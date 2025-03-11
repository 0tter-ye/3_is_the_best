import time
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pybit.unified_trading import HTTP
import logging
import MetaTrader5 as mt5
from imblearn.over_sampling import SMOTE

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from config import API_KEY, API_SECRET  # Ensure config.py exists with mainnet credentials

# Initialize Bybit API (mainnet)
client = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=False)
symbol = "BTCUSDT"

# Initialize MT5 connection
if not mt5.initialize():
    logging.error(f"Failed to initialize MT5: {mt5.last_error()}")
    exit()

mt5_login = 3263156
mt5_password = "Bl1SrRhFb0JP@E4"
mt5_server = "Bybit-Demo"
if not mt5.login(mt5_login, mt5_password, mt5_server):
    logging.error(f"Failed to login to MT5: {mt5.last_error()}")
    exit()
logging.info("Connected to MT5 demo account")

account_info = mt5.account_info()
if account_info is None:
    logging.error("Failed to fetch MT5 account info")
    exit()
initial_balance = account_info.balance
logging.info(f"MT5 Account Balance: {initial_balance} USDT")

symbol_info = mt5.symbol_info(symbol)
if symbol_info is None:
    logging.warning(f"{symbol} not found. Searching for BTCUSDT variant...")
    all_symbols = [s.name for s in mt5.symbols_get() if "BTCUSDT" in s.name]
    if not all_symbols:
        logging.error("No BTCUSDT variant found in MT5. Check Market Watch.")
        exit()
    symbol = all_symbols[0]
    logging.info(f"Using symbol: {symbol}")
    symbol_info = mt5.symbol_info(symbol)

raw_point = symbol_info.point
min_volume = symbol_info.volume_min
max_volume = symbol_info.volume_max
volume_step = symbol_info.volume_step

class ClosureLearner:
    def __init__(self, state_bins, alpha=0.05, gamma=0.95, epsilon=0.1):
        self.state_bins = state_bins
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = ["hold", "close"]

    def discretize_state(self, state):
        total_return, unrealized_pnl, atr, rsi, imbalance, win_rate = state
        bins = self.state_bins
        state_tuple = (
            np.digitize(total_return, bins["total_return"]),
            np.digitize(unrealized_pnl, bins["unrealized_pnl"]),
            np.digitize(atr, bins["atr"]),
            np.digitize(rsi, bins["rsi"]),
            np.digitize(imbalance, bins["imbalance"]),
            np.digitize(win_rate, bins["win_rate"])
        )
        return state_tuple

    def get_q_value(self, state, action):
        state = self.discretize_state(state)
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        state = self.discretize_state(state)
        q_hold = self.q_table.get((state, "hold"), 0.0)
        q_close = self.q_table.get((state, "close"), 0.0)
        logging.debug(f"Choosing action - State: {state}, Q-hold: {q_hold:.2f}, Q-close: {q_close:.2f}")
        return "close" if q_close > q_hold else "hold"

    def update_q_table(self, state, action, reward, next_state):
        state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)
        current_q = self.q_table.get((state, action), 0.0)
        next_max_q = max(self.q_table.get((next_state, a), 0.0) for a in self.actions)
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[(state, action)] = new_q
        logging.debug(f"Q-table update - State: {state}, Action: {action}, Reward: {reward:.2f}, New Q: {new_q:.2f}")

class Wallet:
    def __init__(self, max_risk_per_trade=0.1, max_drawdown=0.1):
        account_info = mt5.account_info()
        self.balance = account_info.balance if account_info else 0
        self.initial_balance = self.balance
        self.positions = {}
        self.trade_history = []
        self.max_risk_per_trade = max_risk_per_trade
        self.max_drawdown = max_drawdown
        self.paused = False
        self.max_profit = {}
        self.manual_override = False
        self.manual_override_time = None
        self.closure_learner = ClosureLearner(
            state_bins={
                "total_return": np.linspace(-20, 20, 11),
                "unrealized_pnl": np.linspace(-10, 10, 11),
                "atr": np.linspace(0, 500, 11),
                "rsi": np.linspace(0, 100, 11),
                "imbalance": np.linspace(-1, 1, 11),
                "win_rate": np.linspace(0, 100, 11)
            }
        )
        self.last_state = None
        self.last_action = None

    def sync_balance(self):
        account_info = mt5.account_info()
        if account_info is not None:
            self.balance = account_info.balance
            logging.debug(f"MT5 Balance: {self.balance}")
            drawdown = (self.initial_balance - self.balance) / self.initial_balance
            if drawdown >= self.max_drawdown:
                self.paused = True
                logging.warning(f"Max drawdown ({self.max_drawdown*100}%) reached. Trading paused.")
            if self.balance < self.initial_balance * 0.95:
                self.close_all_positions()
                self.paused = True
                logging.warning("Equity below 95% of initial balance. All positions closed and trading paused.")
            
            mt5_positions = mt5.positions_get(symbol=symbol)
            current_tickets = {pos.ticket for pos in mt5_positions} if mt5_positions else set()
            wallet_tickets = {pos['ticket'] for pos in self.positions.values() if 'ticket' in pos}
            
            if wallet_tickets and not current_tickets:
                self.manual_override = True
                self.manual_override_time = datetime.now()
                self.paused = True
                self.positions = {}
                logging.info("Manual closure detected. Pausing trading for 1 hour.")
            elif mt5_positions:
                new_positions = {}
                for pos in mt5_positions:
                    if pos.symbol == symbol:
                        entry_price = self.positions.get(symbol, {}).get('entry_price', pos.price_open)
                        new_positions[symbol] = {
                            'ticket': pos.ticket,
                            'qty': pos.volume,
                            'entry_price': entry_price,
                            'side': "Buy" if pos.type == mt5.ORDER_TYPE_BUY else "Sell",
                            'open_time': datetime.fromtimestamp(pos.time),
                            'entry_imbalance': self.positions.get(symbol, {}).get('entry_imbalance', 0),
                            'entry_volume_spike': self.positions.get(symbol, {}).get('entry_volume_spike', 0)
                        }
                self.positions = new_positions
        else:
            logging.warning("Failed to sync balance with MT5")

    def calculate_position_size(self, price, risk_distance, fixed_qty=None, volume_multiplier=1.0):
        if fixed_qty is not None:
            return adjust_volume(fixed_qty, min_volume, max_volume, volume_step)
        risk_amount = self.balance * self.max_risk_per_trade
        qty = (risk_amount / risk_distance) * volume_multiplier
        return adjust_volume(qty, min_volume, max_volume, volume_step)

    def open_position(self, symbol, side, qty, price, df, stop_loss=None, take_profit=None):
        if self.paused:
            logging.warning("Trading paused due to drawdown or manual override.")
            return False
        cost = qty * price
        self.sync_balance()
        if self.balance >= cost:
            self.balance -= cost
            self.positions[symbol] = {
                'qty': qty, 'entry_price': price,
                'side': side,
                'open_time': datetime.now(),
                'entry_imbalance': calculate_bid_ask_imbalance(*fetch_order_book(symbol, df)[1:3]),
                'entry_volume_spike': df["volume_spike_1m"].iloc[-1] if "volume_spike_1m" in df.columns else 0
            }
            self.max_profit[symbol] = 0
            logging.info(f"Opened {side} position: {qty} {symbol} @ {price}")
            return True
        logging.warning(f"Insufficient funds: {cost} > {self.balance}")
        return False

    def calculate_unrealized_pnl(self, symbol, current_price):
        if symbol not in self.positions:
            return 0
        pos = self.positions[symbol]
        qty = pos['qty']
        entry_price = pos['entry_price']
        side = pos['side']
        return qty * (current_price - entry_price) if side == "Buy" else qty * (entry_price - current_price)

    def close_position(self, symbol, price):
        if symbol in self.positions:
            pos = self.positions[symbol]
            qty = pos['qty']
            entry_price = pos['entry_price']
            side = pos['side']
            profit = qty * (price - entry_price) if side == "Buy" else qty * (entry_price - price)
            self.balance += qty * price if side == "Buy" else qty * entry_price + profit
            trade = {
                'symbol': symbol, 'side': side, 'qty': qty,
                'entry_price': entry_price, 'exit_price': price,
                'profit': profit, 'timestamp': datetime.now(),
                'entry_imbalance': pos['entry_imbalance'],
                'entry_volume_spike': pos['entry_volume_spike']
            }
            self.trade_history.append(trade)
            logging.info(f"Closed {side} position: {qty} {symbol} @ {price}, Profit: {profit:.2f}")
            logging.debug(f"Profit calc: qty={qty}, entry={entry_price}, exit={price}, profit={profit:.2f}")
            del self.positions[symbol]
            del self.max_profit[symbol]
            self.sync_balance()

    def close_all_positions(self):
        total_pnl = 0
        for sym in list(self.positions.keys()):
            tick = mt5.symbol_info_tick(sym)
            current_price = tick.ask if self.positions[sym]['side'] == "Buy" else tick.bid
            logging.debug(f"Closing position for {sym} at {current_price}")
            total_pnl += self.calculate_unrealized_pnl(sym, current_price)
            self.close_position(sym, current_price)
        self.positions.clear()
        return total_pnl

    def get_performance_summary(self, trades=None):
        if trades is None:
            trades = self.trade_history
        self.sync_balance()
        total_profit = sum(trade['profit'] for trade in trades)
        total_trades = len(trades)
        final_value = self.initial_balance + total_profit
        total_return = (final_value - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
        win_rate = len([t for t in trades if t['profit'] > 0]) / total_trades * 100 if total_trades > 0 else 0
        return {
            'start_value': self.initial_balance,
            'final_value': final_value,
            'total_return': total_return * 100,
            'total_trades': total_trades,
            'win_rate': win_rate
        }

    def get_current_state(self, df, imbalance):
        summary = self.get_performance_summary()
        total_return = summary['total_return']
        unrealized_pnl = sum(self.calculate_unrealized_pnl(sym, df["close_5m"].iloc[-1]) for sym in self.positions) / self.initial_balance * 100
        atr = df["atr_5m"].iloc[-1]
        rsi = df["rsi_5m"].iloc[-1]
        win_rate = summary['win_rate']
        return (total_return, unrealized_pnl, atr, rsi, imbalance, win_rate)

wallet = Wallet(max_risk_per_trade=0.01)

class MarkovChain:
    def __init__(self, states, transition_matrix):
        self.states = states
        self.transition_matrix = transition_matrix
        self.current_state = random.choice(states)
        self.stationary_dist = self.compute_stationary_distribution()

    def compute_stationary_distribution(self):
        P = np.array(self.transition_matrix)
        n = P.shape[0]
        A = P.T - np.eye(n)
        A[-1] = np.ones(n)
        b = np.zeros(n)
        b[-1] = 1
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
        pi = np.clip(pi, 0, None)
        pi /= pi.sum()
        return pi.tolist()

    def next_state(self):
        try:
            probabilities = self.transition_matrix[self.states.index(self.current_state)]
            probabilities = np.clip(probabilities, 0, None)
            prob_sum = np.sum(probabilities)
            if prob_sum == 0:
                probabilities = np.ones_like(probabilities) / len(probabilities)
            else:
                probabilities = probabilities / prob_sum
            self.current_state = np.random.choice(self.states, p=probabilities)
            return self.current_state, self.stationary_dist
        except Exception as e:
            logging.error(f"Error in Markov Chain transition: {e}")
            self.current_state = random.choice(self.states)
            return self.current_state, [0.5, 0.5]

states = ["Loss", "Win"]
transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=float)
markov_chain = MarkovChain(states, transition_matrix)
logging.info(f"Markov Stationary Dist: {markov_chain.stationary_dist}")

def fetch_data(symbol, timeframe, limit=200):
    try:
        now = int(time.time() * 1000)
        response = client.get_kline(category="linear", symbol=symbol, interval=timeframe, limit=limit, start=now - limit * int(timeframe) * 60000)
        if "result" not in response or "list" not in response["result"]:
            logging.error(f"Invalid API response structure for {timeframe} timeframe")
            return pd.DataFrame()
        
        data = response["result"]["list"]
        logging.debug(f"Raw API response for {timeframe}m (last 5): {data[:5]}")
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms")
        df[["open", "high", "low", "close", "volume", "turnover"]] = df[["open", "high", "low", "close", "volume", "turnover"]].astype(float)
        
        # Initial imputation
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        df[["open", "high", "low", "close", "volume", "turnover"]] = imputer.fit_transform(df[["open", "high", "low", "close", "volume", "turnover"]])
        
        # Feature engineering
        df["returns"] = df["close"].pct_change()
        df["high_low"] = df["high"] - df["low"]
        df["high_close_prev"] = abs(df["high"] - df["close"].shift(1))
        df["low_close_prev"] = abs(df["low"] - df["close"].shift(1))
        df["tr"] = df[["high_low", "high_close_prev", "low_close_prev"]].max(axis=1)
        df["atr"] = df["tr"].rolling(window=14).mean()
        df["volume_mean"] = df["volume"].rolling(window=20).mean()
        df["volume_std"] = df["volume"].rolling(window=20).std()
        df["volume_spike"] = (df["volume"] > df["volume_mean"] + 2 * df["volume_std"]).astype(int)
        
        # Final imputation for all numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        df = df.drop(columns=["high_low", "high_close_prev", "low_close_prev", "tr"])
        df.columns = [f"{col}_{timeframe}m" if col not in ["timestamp"] else col for col in df.columns]
        df = df.sort_values("timestamp")
        logging.debug(f"Latest timestamp for {timeframe}m: {df['timestamp'].iloc[-1]}")
        return df
    except Exception as e:
        logging.error(f"Error fetching {timeframe}min data: {e}")
        return pd.DataFrame()

def calculate_sma(df, period, timeframe="5m"):
    return df[f"close_{timeframe}"].rolling(window=period).mean()

def calculate_rsi(df, period=14, timeframe="5m"):
    delta = df[f"close_{timeframe}"].diff()
    logging.debug(f"Delta sample: {delta.tail().tolist()}")
    if delta.abs().sum() == 0:
        logging.warning(f"No price variation in {timeframe}m data. Setting RSI to 50.")
        return pd.Series(50, index=df.index)
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    logging.debug(f"Gain: {gain.iloc[-1]:.2f}, Loss: {loss.iloc[-1]:.2f}")
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    logging.debug(f"RSI last value: {rsi.iloc[-1]:.2f}")
    return rsi

def is_trend_reversal(df, short_period=10, long_period=30, timeframe="5m"):
    sma_short = calculate_sma(df, short_period, timeframe)
    sma_long = calculate_sma(df, long_period, timeframe)
    rsi = calculate_rsi(df, timeframe="5m")
    
    last_sma_short = sma_short.iloc[-1]
    last_sma_long = sma_long.iloc[-1]
    prev_sma_short = sma_short.iloc[-2]
    prev_sma_long = sma_long.iloc[-2]
    last_rsi = rsi.iloc[-1]
    
    bullish_cross = prev_sma_short <= prev_sma_long and last_sma_short > last_sma_long and last_rsi > 50
    bearish_cross = prev_sma_short >= prev_sma_long and last_sma_short < last_sma_long and last_rsi < 50
    return bullish_cross, bearish_cross

def calculate_support_resistance(df, window=20, timeframe="5m"):
    df = df.copy()
    if len(df) < window:
        logging.warning(f"Insufficient data for support/resistance: {len(df)} rows < window {window}")
        return [], []
    
    df["pivot_high"] = df[f"high_{timeframe}"].rolling(window=window, min_periods=1).max().shift(1)
    df["pivot_low"] = df[f"low_{timeframe}"].rolling(window=window, min_periods=1).min().shift(1)
    
    resistance_levels = df[df[f"high_{timeframe}"] == df["pivot_high"]][f"high_{timeframe}"].dropna().value_counts()
    resistance_levels = resistance_levels[resistance_levels >= 1].index.tolist()[:3]
    
    support_levels = df[df[f"low_{timeframe}"] == df["pivot_low"]][f"low_{timeframe}"].dropna().value_counts()
    support_levels = support_levels[support_levels >= 1].index.tolist()[:3]
    
    logging.debug(f"Support candidates: {support_levels}, Resistance candidates: {resistance_levels}")
    return sorted(support_levels), sorted(resistance_levels, reverse=True)

def fetch_combined_data(symbol, timeframes=["1", "3", "5"], limit=500):
    dfs = {}
    for tf in timeframes:
        df = fetch_data(symbol, tf, limit)
        if not df.empty:
            dfs[tf] = df
        else:
            logging.warning(f"Failed to fetch {tf}min data")
    
    if not dfs:
        logging.error("No data fetched for any timeframe.")
        return pd.DataFrame()
    
    base_tf = min(timeframes, key=int)
    combined_df = dfs[base_tf]
    for tf in timeframes:
        if tf != base_tf:
            combined_df = pd.merge_asof(
                combined_df,
                dfs[tf],
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta(minutes=int(tf)),
                suffixes=("", f"_{tf}m")
            )
    combined_df["sma_short_5m"] = calculate_sma(combined_df, 10, "5m")
    combined_df["sma_long_5m"] = calculate_sma(combined_df, 30, "5m")
    combined_df["rsi_5m"] = calculate_rsi(combined_df, 14, "5m")
    
    # Final imputation for all numerical columns
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    combined_df[numeric_cols] = imputer.fit_transform(combined_df[numeric_cols])
    
    # Check for recent price variation
    recent_5m = combined_df["close_5m"].tail(5)
    if recent_5m.std() < 1e-6:
        logging.warning(f"Recent 5m prices static: {recent_5m.tolist()}. Check API data source.")
    
    logging.info(f"Combined data shape: {combined_df.shape}, Timeframes: {timeframes}")
    logging.debug(f"Price range 5m: {combined_df['close_5m'].max() - combined_df['close_5m'].min():.2f}")
    return combined_df

def fetch_order_book(symbol, df, depth=10):
    try:
        response = client.get_orderbook(category="linear", symbol=symbol, limit=depth)
        if "result" not in response or "b" not in response["result"] or "a" not in response["result"]:
            logging.warning(f"Failed to fetch order book for {symbol}. Using fallback.")
            return [100.0], [10], [100.1], [10]
        
        bids = response["result"]["b"]
        asks = response["result"]["a"]
        bid_prices = [float(b[0]) for b in bids]
        bid_volumes = [float(b[1]) for b in bids]
        ask_prices = [float(a[0]) for a in asks]
        ask_volumes = [float(a[1]) for a in asks]
        
        current_price = (max(bid_prices) + min(ask_prices)) / 2
        last_close = df["close_5m"].iloc[-1] if not df.empty else None
        if last_close and abs(current_price - last_close) / last_close > 0.5:
            logging.warning(f"Order book price {current_price} deviates significantly from close {last_close}. Using close price.")
            current_price = last_close
        
        logging.debug(f"Order book fetched: {len(bid_prices)} bids, {len(ask_prices)} asks")
        return bid_prices, bid_volumes, ask_prices, ask_volumes
    except Exception as e:
        logging.error(f"Error fetching order book: {e}")
        return [100.0], [10], [100.1], [10]

def calculate_bid_ask_imbalance(bid_volumes, ask_volumes):
    total_bid_volume = min(sum(bid_volumes), 10000)
    total_ask_volume = min(sum(ask_volumes), 10000)
    total_bid_volume = max(total_bid_volume, 1)
    total_ask_volume = max(total_ask_volume, 1)
    logging.debug(f"Bid Volume: {total_bid_volume}, Ask Volume: {total_ask_volume}")
    total = total_bid_volume + total_ask_volume
    return (total_bid_volume - total_ask_volume) / total

def adjust_volume(volume, min_vol, max_vol, step):
    volume = max(min_vol, min(max_vol, volume))
    volume = round(volume / step) * step
    return round(volume, 6)

def train_models(df):
    if df.empty or not any(col.endswith("_5m") for col in df.columns):
        logging.error("DataFrame is empty or missing required 5m columns")
        return None, None, None
    
    if df["close_5m"].std() < 1e-6:
        logging.warning("Price variation too low for meaningful training. Skipping retrain.")
        return rf_model, dt_model, scaler
    
    bid_prices, bid_volumes, ask_prices, ask_volumes = fetch_order_book(symbol, df)
    df["imbalance"] = calculate_bid_ask_imbalance(bid_volumes, ask_volumes)
    df["trend_5m"] = (df["close_5m"] - df["close_5m"].shift(5)) / df["atr_5m"]
    feature_cols = [col for col in df.columns if col not in ["timestamp"] and not col.startswith("turnover") and "5m" in col] + ["imbalance", "trend_5m"]
    X = df[feature_cols].iloc[:-1]
    y = (df["close_5m"].shift(-1) > df["close_5m"] + df["atr_5m"].iloc[:-1] * 0.25).astype(int).iloc[:-1]
    if len(y) < 10:
        logging.warning("Not enough data for training")
        return None, None, None
    
    if X.isna().any().any():
        logging.warning(f"NaNs detected in X before imputation: {X.isna().sum()}")
    
    # Impute NaN values before SMOTE
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imputed = imputer.fit_transform(X)
    logging.info(f"Imputation completed. NaNs after imputation: {np.isnan(X_imputed).sum()}")

    logging.info(f"Training data class distribution: {np.bincount(y)}")
    minority_count = np.bincount(y)[1]
    if minority_count < 2:
        logging.warning("Too few positive samples for meaningful training. Skipping SMOTE and using raw data.")
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    else:
        n_neighbors = min(5, max(1, minority_count - 1))
        smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
        try:
            X_resampled, y_resampled = smote.fit_resample(X_imputed, y)
            logging.info(f"Post-SMOTE class distribution: {np.bincount(y_resampled)}")
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        except ValueError as e:
            logging.warning(f"SMOTE failed: {e}. Falling back to raw data.")
            X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=10, random_state=42, class_weight="balanced")
    rf_model.fit(X_train_scaled, y_train)
    rf_accuracy = rf_model.score(X_test_scaled, y_test)

    dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42, class_weight="balanced")
    dt_model.fit(X_train_scaled, y_train)
    dt_accuracy = dt_model.score(X_test_scaled, y_test)

    logging.info(f"RF Model Accuracy: {rf_accuracy:.2f}")
    logging.info(f"DT Model Accuracy: {dt_accuracy:.2f}, Max Depth: {dt_model.get_depth()}, Leaves: {dt_model.get_n_leaves()}")
    return rf_model, dt_model, scaler

rf_model, dt_model, scaler = None, None, None

# Initial data fetch with retry
max_retries = 3
for attempt in range(max_retries):
    df_initial = fetch_combined_data(symbol, timeframes=["1", "3", "5"], limit=1000)
    if not df_initial.empty:
        break
    logging.warning(f"Initial data fetch attempt {attempt + 1} failed. Retrying in 15 seconds...")
    time.sleep(15)
else:
    logging.error("Initial data fetch failed after all retries. Exiting.")
    exit()

rf_model, dt_model, scaler = train_models(df_initial)
if rf_model is None or dt_model is None or scaler is None:
    logging.error("Model training failed. Exiting.")
    exit()

def execute_trade(prediction, symbol, df, confidence_threshold=0.50, markov_win_prob=0.0):
    if wallet.paused:
        logging.warning("Trading paused.")
        return
    
    side = "Buy" if prediction == 1 else "Sell"
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logging.error("Failed to fetch MT5 tick data.")
            return
        current_price = tick.ask if side == "Buy" else tick.bid
        logging.debug(f"Using MT5 price: {current_price}")

        atr_5m = df["atr_5m"].iloc[-1]
        volume_spike_1m = df["volume_spike_1m"].iloc[-1] if "volume_spike_1m" in df.columns else 0
        feature_cols = [col for col in df.columns if col not in ["timestamp"] and not col.startswith("turnover") and "5m" in col] + ["imbalance", "trend_5m"]
        X_latest = pd.DataFrame(df[feature_cols].iloc[-1]).T
        logging.debug(f"X_latest before scaling: {X_latest.values[0][:5]}")
        X_latest_scaled = scaler.transform(X_latest)
        rf_confidence = rf_model.predict_proba(X_latest_scaled)[0][1] if prediction == 1 else 1 - rf_model.predict_proba(X_latest_scaled)[0][1]
        dt_confidence = dt_model.predict_proba(X_latest_scaled)[0][1] if prediction == 1 else 1 - rf_model.predict_proba(X_latest_scaled)[0][1]
        
        logging.debug(f"RF Confidence: {rf_confidence:.2f}, DT Confidence: {dt_confidence:.2f}, Threshold: {confidence_threshold}")
        
        if (prediction == 1 and (rf_confidence < confidence_threshold or dt_confidence < confidence_threshold)) or \
           (prediction == 0 and (rf_confidence < confidence_threshold or dt_confidence < confidence_threshold)):
            logging.debug(f"Models disagree or below threshold: RF={rf_confidence:.2f}, DT={dt_confidence:.2f}")
            return
        
        bid_prices, bid_volumes, ask_prices, ask_volumes = fetch_order_book(symbol, df, depth=10)
        imbalance = calculate_bid_ask_imbalance(bid_volumes, ask_volumes)
        big_move_factor = 1.0
        if volume_spike_1m and imbalance > 0.3 and side == "Buy":
            big_move_factor = 1.2
        elif volume_spike_1m and imbalance < -0.3 and side == "Sell":
            big_move_factor = 1.2
        
        total_score = (0.35 * max(0, (rf_confidence - confidence_threshold) / (1 - confidence_threshold)) +
                       0.15 * max(0, (dt_confidence - confidence_threshold) / (1 - confidence_threshold)) +
                       0.2 * max(0, (markov_win_prob / 100 - 0.60) / 0.40)) * big_move_factor
        total_score = np.clip(total_score, 0, 1)
        
        volume_range = [0.01, 0.02, 0.03, 0.04, 0.05]
        trade_qty = volume_range[min(int(total_score * (len(volume_range) - 1)), len(volume_range) - 1)]
        adjusted_qty = wallet.calculate_position_size(current_price, atr_5m * 8, fixed_qty=trade_qty)

        if adjusted_qty < min_volume:
            logging.warning(f"Trade qty {adjusted_qty} below minimum {min_volume}. Skipping.")
            return

        cost = adjusted_qty * current_price
        wallet.sync_balance()
        if cost > wallet.balance:
            logging.warning(f"Trade cost {cost:.2f} USDT exceeds balance {wallet.balance:.2f} USDT. Skipping.")
            return

        order_type = mt5.ORDER_TYPE_BUY if side == "Buy" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": adjusted_qty,
            "type": order_type,
            "price": current_price,
            "deviation": 20,
            "magic": 123456,
            "comment": f"{side} via Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order failed: {result.comment} (retcode: {result.retcode})")
            return
        
        logging.info(f"MT5 Order Success: Ticket={result.order}, Price={result.price}, Volume={result.volume}")
        if wallet.open_position(symbol, side, adjusted_qty, current_price, df):
            wallet.positions[symbol]['ticket'] = result.order
            logging.info(f"Placed {side} order: {adjusted_qty} {symbol} @ {current_price}, Imbalance: {imbalance:.2f}, Volume Spike: {volume_spike_1m}")
        else:
            logging.error("Failed to open position in wallet after successful order.")

    except Exception as e:
        logging.error(f"Error executing trade: {str(e)}")

def main_loop():
    global rf_model, dt_model, scaler
    iteration = 0
    start_time = datetime.now()
    last_retrain = datetime.now()
    retrain_interval = timedelta(hours=1)
    atr_threshold = None

    while True:
        try:
            df = fetch_combined_data(symbol, timeframes=["1", "3", "5"], limit=500)
            if df.empty:
                logging.warning("No valid data fetched. Retrying in 15 seconds...")
                time.sleep(15)
                continue

            logging.debug(f"Iteration {iteration} - 5m Close Prices (last 5): {df['close_5m'].tail().tolist()}")

            if atr_threshold is None or iteration % 10 == 0:
                atr_mean = df["atr_5m"].rolling(window=14).mean().iloc[-1]
                atr_threshold = atr_mean * 2
            if df["atr_5m"].iloc[-1] > atr_threshold:
                logging.info(f"ATR {df['atr_5m'].iloc[-1]:.2f} exceeds threshold {atr_threshold:.2f}. Pausing iteration.")
                time.sleep(15)
                continue

            logging.debug(f"Iteration {iteration} - Data columns: {df.columns.tolist()}")

            if datetime.now() - last_retrain >= retrain_interval:
                new_rf_model, new_dt_model, new_scaler = train_models(df)
                if new_rf_model is not None and new_dt_model is not None and new_scaler is not None:
                    rf_model, dt_model, scaler = new_rf_model, new_dt_model, new_scaler
                last_retrain = datetime.now()

            if wallet.paused and wallet.manual_override:
                if (datetime.now() - wallet.manual_override_time) >= timedelta(hours=1):
                    wallet.paused = False
                    wallet.manual_override = False
                    logging.info("Resuming trading after 1-hour manual override pause.")
                else:
                    logging.info("Trading paused due to manual override.")
                    time.sleep(15)
                    continue

            current_state, stationary_dist = markov_chain.next_state()
            markov_win_prob = stationary_dist[1] * 100
            logging.debug(f"Iteration {iteration} - Markov State: {current_state}, Win Prob: {markov_win_prob:.2f}%")
            
            bid_prices, bid_volumes, ask_prices, ask_volumes = fetch_order_book(symbol, df, depth=10)
            imbalance = calculate_bid_ask_imbalance(bid_volumes, ask_volumes)
            df["imbalance"] = imbalance
            df["trend_5m"] = (df["close_5m"] - df["close_5m"].shift(5)) / df["atr_5m"]

            feature_cols = [col for col in df.columns if col not in ["timestamp"] and not col.startswith("turnover") and "5m" in col] + ["imbalance", "trend_5m"]
            X_latest = pd.DataFrame(df[feature_cols].iloc[-1]).T
            X_latest_scaled = scaler.transform(X_latest)
            
            rf_pred = rf_model.predict_proba(X_latest_scaled)[0][1]
            dt_pred = dt_model.predict_proba(X_latest_scaled)[0][1]
            logging.debug(f"Iteration {iteration} - RF Pred: {rf_pred:.2f}, DT Pred: {dt_pred:.2f}")
            logging.debug(f"Iteration {iteration} - Features: close_5m={df['close_5m'].iloc[-1]}, rsi_5m={df['rsi_5m'].iloc[-1]}, atr_5m={df['atr_5m'].iloc[-1]}, imbalance={imbalance}, trend_5m={df['trend_5m'].iloc[-1]}")
            
            if iteration == 1:
                prediction = 1 if rf_pred > 0.5 else 0
                logging.info(f"Iteration {iteration} - Forcing test trade: RF Pred: {rf_pred:.2f}, DT Pred: {dt_pred:.2f}, Prediction: {prediction}")
                execute_trade(prediction, symbol, df, confidence_threshold=0.50, markov_win_prob=markov_win_prob)
            else:
                trend_strength = (df["close_5m"].iloc[-1] - df["close_5m"].iloc[-5]) / df["atr_5m"].iloc[-1]
                adjusted_buy_threshold = max(0.5, 0.7 - trend_strength * 0.1)
                adjusted_sell_threshold = 0.5
                if rf_pred > adjusted_buy_threshold and dt_pred > adjusted_buy_threshold:
                    prediction = 1
                elif rf_pred < adjusted_sell_threshold and dt_pred < adjusted_sell_threshold:
                    prediction = 0
                else:
                    prediction = None
                logging.info(f"Iteration {iteration} - RF Win Prob: {rf_pred*100:.2f}%, DT Win Prob: {dt_pred*100:.2f}%, "
                             f"Prediction: {prediction}, Buy Threshold: {adjusted_buy_threshold:.2f}, Sell Threshold: {adjusted_sell_threshold:.2f}")
                if prediction is not None:
                    execute_trade(prediction, symbol, df, confidence_threshold=min(adjusted_buy_threshold, adjusted_sell_threshold), markov_win_prob=markov_win_prob)

            tick = mt5.symbol_info_tick(symbol)
            current_price_mt5 = tick.ask if wallet.positions.get(symbol, {}).get('side') == "Buy" else tick.bid
            bybit_price = (min(ask_prices) + max(bid_prices)) / 2
            logging.debug(f"Iteration {iteration} - MT5 Price: {current_price_mt5}, Bybit Price: {bybit_price}, Selected Price: {current_price_mt5}")
            
            imbalance = calculate_bid_ask_imbalance(bid_volumes, ask_volumes)
            
            wallet_summary = wallet.get_performance_summary()
            total_return = wallet_summary["total_return"]
            base_profit_target = 3.0
            atr_5m = df["atr_5m"].iloc[-1]
            rsi_5m = df["rsi_5m"].iloc[-1]
            
            volatility_factor = atr_5m / df["close_5m"].iloc[-1] * 100
            dynamic_profit_target = base_profit_target * (1 + volatility_factor / 100)
            if total_return > 5:
                dynamic_profit_target += (total_return / 10)
                logging.info(f"Wallet up {total_return:.2f}%, Volatility-Adjusted Profit Target: {dynamic_profit_target:.2f}%")
            elif total_return < -5:
                dynamic_profit_target = max(1.0, base_profit_target - 1.0)
                logging.info(f"Wallet down {total_return:.2f}%, Reduced Profit Target: {dynamic_profit_target:.2f}%")

            bullish_reversal, bearish_reversal = is_trend_reversal(df)
            support_levels, resistance_levels = calculate_support_resistance(df, window=20, timeframe="5m")
            logging.debug(f"Iteration {iteration} - Support Levels: {support_levels}, Resistance Levels: {resistance_levels}")

            current_state = wallet.get_current_state(df, imbalance)
            positions_closed_by_rl = False
            if wallet.positions:
                action = wallet.closure_learner.choose_action(current_state)
                if action == "close":
                    total_pnl = wallet.close_all_positions()
                    logging.info(f"Iteration {iteration} - RL Closure triggered. Total PNL: {total_pnl:.2f}")
                    reward = total_pnl / wallet.initial_balance * 100
                    next_state = wallet.get_current_state(df, imbalance)
                    if wallet.last_state and wallet.last_action:
                        wallet.closure_learner.update_q_table(wallet.last_state, wallet.last_action, reward, next_state)
                    positions_closed_by_rl = True
                else:
                    reward = sum(wallet.calculate_unrealized_pnl(sym, current_price_mt5) for sym in wallet.positions) / wallet.initial_balance * 100
                    if wallet.last_state and wallet.last_action:
                        next_state = current_state
                        wallet.closure_learner.update_q_table(wallet.last_state, wallet.last_action, reward, next_state)
                wallet.last_state = current_state
                wallet.last_action = action

            if not positions_closed_by_rl:
                try:
                    for sym, pos in list(wallet.positions.items()):
                        if sym not in wallet.positions:
                            continue
                        unrealized_pnl = wallet.calculate_unrealized_pnl(sym, current_price_mt5)
                        entry_price = pos["entry_price"]
                        side = pos["side"]
                        time_open = (datetime.now() - pos["open_time"]).total_seconds() / 60

                        profit_percent = (unrealized_pnl / (pos["qty"] * entry_price)) * 100
                        loss_percent = -profit_percent if profit_percent < 0 else 0

                        wallet.max_profit[sym] = max(wallet.max_profit[sym], profit_percent)
                        trailing_stop = wallet.max_profit[sym] - 1.0

                        near_resistance = any(abs(current_price_mt5 - r) / current_price_mt5 < 0.005 for r in resistance_levels)
                        near_support = any(abs(current_price_mt5 - s) / current_price_mt5 < 0.005 for s in support_levels)
                        
                        profit_target = 1.5 if side == "Buy" else dynamic_profit_target
                        
                        if side == "Buy" and near_resistance and (imbalance < -0.2 or profit_percent > 0):
                            logging.info(f"Iteration {iteration} - Near resistance {resistance_levels}, Imbalance: {imbalance:.2f}, closing Buy with profit: {profit_percent:.2f}%")
                            wallet.close_position(sym, current_price_mt5)
                        elif side == "Sell" and near_support and (imbalance > 0.2 or profit_percent > 0):
                            logging.info(f"Iteration {iteration} - Near support {support_levels}, Imbalance: {imbalance:.2f}, closing Sell with profit: {profit_percent:.2f}%")
                            wallet.close_position(sym, current_price_mt5)
                        elif profit_percent >= profit_target:
                            logging.info(f"Iteration {iteration} - Profit target hit: {profit_percent:.2f}% (Target: {profit_target:.2f}%)")
                            wallet.close_position(sym, current_price_mt5)
                        elif profit_percent >= 1.5 and rf_model.predict_proba(X_latest_scaled)[0][1 if side == "Buy" else 0] < 0.5:
                            logging.info(f"Iteration {iteration} - Profit secured due to reversal signal: {profit_percent:.2f}%")
                            wallet.close_position(sym, current_price_mt5)
                        elif side == "Buy" and rsi_5m > 70 and profit_percent > 0:
                            logging.info(f"Iteration {iteration} - Overbought (RSI: {rsi_5m:.2f}), closing Buy with profit: {profit_percent:.2f}%")
                            wallet.close_position(sym, current_price_mt5)
                        elif side == "Sell" and rsi_5m < 30 and rsi_5m > 0 and profit_percent > 0:
                            logging.info(f"Iteration {iteration} - Oversold (RSI: {rsi_5m:.2f}), closing Sell with profit: {profit_percent:.2f}%")
                            wallet.close_position(sym, current_price_mt5)
                        elif profit_percent > 2.0 and profit_percent < trailing_stop:
                            logging.info(f"Iteration {iteration} - Trailing stop triggered: {profit_percent:.2f}% (Max: {wallet.max_profit[sym]:.2f}%)")
                            wallet.close_position(sym, current_price_mt5)
                        elif side == "Buy" and bearish_reversal and profit_percent > 0:
                            logging.info(f"Iteration {iteration} - Bearish reversal detected, closing Buy with profit: {profit_percent:.2f}%")
                            wallet.close_position(sym, current_price_mt5)
                        elif side == "Sell" and bullish_reversal and profit_percent > 0:
                            logging.info(f"Iteration {iteration} - Bullish reversal detected, closing Sell with profit: {profit_percent:.2f}%")
                            wallet.close_position(sym, current_price_mt5)
                        elif side == "Buy" and imbalance < -0.5 and df["volume_spike_1m"].iloc[-1]:
                            logging.info(f"Iteration {iteration} - Big sell detected, closing Buy: Imbalance {imbalance:.2f}")
                            wallet.close_position(sym, current_price_mt5)
                        elif side == "Sell" and imbalance > 0.5 and df["volume_spike_1m"].iloc[-1]:
                            logging.info(f"Iteration {iteration} - Big buy detected, closing Sell: Imbalance {imbalance:.2f}")
                            wallet.close_position(sym, current_price_mt5)
                        elif loss_percent >= 2.0:
                            logging.info(f"Iteration {iteration} - Loss limit hit: -{loss_percent:.2f}%")
                            wallet.close_position(sym, current_price_mt5)
                        elif time_open >= 10:
                            logging.info(f"Iteration {iteration} - Time limit hit: {time_open:.1f} minutes, Profit: {profit_percent:.2f}%")
                            wallet.close_position(sym, current_price_mt5)
                except Exception as e:
                    logging.error(f"Iteration {iteration} - Error in position management: {repr(e)}")

            iteration += 1
            if iteration % 10 == 0:
                summary = wallet.get_performance_summary()
                logging.info(f"Iteration {iteration} - Overall Performance: Start Value: {summary['start_value']:.2f}, "
                             f"Final Value: {summary['final_value']:.2f}, Total Return: {summary['total_return']:.2f}%, "
                             f"Trades: {summary['total_trades']}, Win Rate: {summary['win_rate']:.2f}%")

            time.sleep(15)
        except Exception as e:
            logging.error(f"Iteration {iteration} - Unexpected error in main loop: {repr(e)}")
            time.sleep(15)

if __name__ == "__main__":
    simulated_history = [
        {"total_return": 10.0, "unrealized_pnl": 8.0, "atr": 200, "rsi": 60, "imbalance": -0.3, "win_rate": 80, "action": "close", "reward": 10.0},
        {"total_return": 2.0, "unrealized_pnl": 1.0, "atr": 100, "rsi": 50, "imbalance": 0.1, "win_rate": 60, "action": "hold", "reward": 0.5},
    ]
    for event in simulated_history:
        state = (event["total_return"], event["unrealized_pnl"], event["atr"], event["rsi"], event["imbalance"], event["win_rate"])
        next_state = state
        wallet.closure_learner.update_q_table(state, event["action"], event["reward"], next_state)
    main_loop()