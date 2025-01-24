from typing import Dict, List
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from dataclasses import dataclass

class BaseFeatures:
    """Base class for feature generation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data"""
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
class PriceFeatures(BaseFeatures):
    """Generate price-based features"""
    
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate price features"""
        
        self._validate_data(data)
        features = pd.DataFrame(index=data.index)
        
        # Log prices and ratios
        epsilon = 1e-10
        features['log_close'] = np.log(data['Close'].clip(lower=epsilon))
        features['log_volume'] = np.log(data['Volume'].clip(lower=epsilon))
        features['high_low_ratio'] = data['High'] / data['Low'].clip(lower=epsilon)
        features['close_open_ratio'] = data['Close'] / data['Open'].clip(lower=epsilon)
        
        # Moving averages - Calculate all MAs first
        ma_windows = [5, 10, 21, 50]
        for window in ma_windows:
            features[f'ma_{window}'] = data['Close'].rolling(window=window).mean()
        
        # MA crossovers - Now we can safely calculate crossovers
        for window in ma_windows[:-1]:  # Exclude 50 as it's our reference
            features[f'ma_{window}_cross_50'] = (
                features[f'ma_{window}'] > features['ma_50']
            ).astype(float)
        
        # Price channels
        for window in [10, 20]:
            features[f'upper_channel_{window}'] = data['High'].rolling(window).max()
            features[f'lower_channel_{window}'] = data['Low'].rolling(window).min()
            features[f'channel_position_{window}'] = (
                (data['Close'] - features[f'lower_channel_{window}']) /
                (features[f'upper_channel_{window}'] - features[f'lower_channel_{window}'])
            ).clip(0, 1)
        
        # Momentum and acceleration
        features['price_momentum'] = data['Close'].pct_change(5)
        features['price_acceleration'] = features['price_momentum'].diff(5)
        
        # Gap analysis
        features['gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        features['gap_ma'] = features['gap'].rolling(10).mean()
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        return [
            'log_close', 'log_volume', 'high_low_ratio', 'close_open_ratio',
            'ma_5', 'ma_10', 'ma_21', 'ma_50',
            'ma_5_cross_50', 'ma_10_cross_50', 'ma_21_cross_50',
            'upper_channel_10', 'lower_channel_10', 'channel_position_10',
            'upper_channel_20', 'lower_channel_20', 'channel_position_20',
            'price_momentum', 'price_acceleration', 'gap', 'gap_ma'
        ]

class ReturnFeatures(BaseFeatures):
    """Generate return-based features"""
    
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate return features"""
        self._validate_data(data)
        features = pd.DataFrame(index=data.index)
        
        # Multi-timeframe returns
        for window in [1, 3, 5, 10, 21]:
            features[f'return_{window}d'] = data['Close'].pct_change(window)
            features[f'return_{window}d_std'] = (
                features[f'return_{window}d'].rolling(window).std()
            )
        
        # Return momentum
        for window in [5, 10, 21]:
            features[f'return_momentum_{window}d'] = (
                features['return_1d'].rolling(window).mean()
            )
        
        # Drawdown metrics
        rolling_max = data['Close'].rolling(window=252, min_periods=1).max()
        drawdown = (data['Close'] - rolling_max) / rolling_max
        features['drawdown'] = drawdown
        features['drawdown_duration'] = (
            (drawdown == 0).astype(int).groupby(
                (drawdown == 0).astype(int).cumsum()
            ).cumsum()
        )
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        return [
            'return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_21d',
            'return_1d_std', 'return_3d_std', 'return_5d_std', 'return_10d_std', 'return_21d_std',
            'return_momentum_5d', 'return_momentum_10d', 'return_momentum_21d',
            'drawdown', 'drawdown_duration'
        ]

class MomentumFeatures(BaseFeatures):
    """Generate momentum-based features"""
    
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum features"""
        self._validate_data(data)
        features = pd.DataFrame(index=data.index)
        
        # RSI at multiple timeframes
        for window in [6, 14, 28]:
            features[f'rsi_{window}'] = ta.rsi(data['Close'], length=window)
        
        # MACD
        macd = ta.macd(data['Close'])
        features['macd'] = macd['MACD_12_26_9']
        features['macd_signal'] = macd['MACDs_12_26_9']
        features['macd_hist'] = macd['MACDh_12_26_9']
        
        # Stochastic oscillator
        stoch = ta.stoch(data['High'], data['Low'], data['Close'])
        features['stoch_k'] = stoch['STOCHk_14_3_3']
        features['stoch_d'] = stoch['STOCHd_14_3_3']
        
        # ADX
        adx = ta.adx(data['High'], data['Low'], data['Close'])
        features['adx'] = adx['ADX_14']
        features['di_plus'] = adx['DMP_14']
        features['di_minus'] = adx['DMN_14']
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        return [
            'rsi_6', 'rsi_14', 'rsi_28',
            'macd', 'macd_signal', 'macd_hist',
            'stoch_k', 'stoch_d',
            'adx', 'di_plus', 'di_minus'
        ]

class VolatilityFeatures(BaseFeatures):
    """Generate volatility-based features"""
    
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility features"""
        self._validate_data(data)
        features = pd.DataFrame(index=data.index)
        
        # ATR and variations
        atr = ta.atr(data['High'], data['Low'], data['Close'])
        features['atr'] = atr
        features['atr_percent'] = atr / data['Close']
        
        # Bollinger Bands
        for window in [20, 40]:
            bb = ta.bbands(data['Close'], length=window)
            features[f'bb_upper_{window}'] = bb[f'BBU_{window}_2.0']
            features[f'bb_lower_{window}'] = bb[f'BBL_{window}_2.0']
            features[f'bb_width_{window}'] = (
                (bb[f'BBU_{window}_2.0'] - bb[f'BBL_{window}_2.0']) /
                bb[f'BBM_{window}_2.0']
            )
        
        # Historical volatility
        returns = data['Close'].pct_change()
        for window in [5, 21, 63]:
            features[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Volatility regime
        features['volatility_regime'] = (
            features['volatility_21d'] > features['volatility_21d'].rolling(252).mean()
        ).astype(float)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        return [
            'atr', 'atr_percent',
            'bb_upper_20', 'bb_lower_20', 'bb_width_20',
            'bb_upper_40', 'bb_lower_40', 'bb_width_40',
            'volatility_5d', 'volatility_21d', 'volatility_63d',
            'volatility_regime'
        ]

class VolumeFeatures(BaseFeatures):
    """Generate volume-based features"""
    
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume features"""
        self._validate_data(data)
        features = pd.DataFrame(index=data.index)
        
        # Volume trends
        for window in [5, 10, 21]:
            features[f'volume_ma_{window}'] = data['Volume'].rolling(window).mean()
            features[f'volume_ratio_{window}'] = (
                data['Volume'] / features[f'volume_ma_{window}']
            )
        
        # OBV and variations
        features['obv'] = ta.obv(data['Close'], data['Volume'])
        features['obv_ma'] = features['obv'].rolling(21).mean()
        features['obv_trend'] = features['obv'] > features['obv_ma']
        
        # Volume price trend
        features['volume_price_trend'] = (
            data['Volume'] * (data['Close'] - data['Open']) / data['Open']
        )
        features['vpt_ma'] = features['volume_price_trend'].rolling(21).mean()
        
        # Volume breakouts
        volume_std = data['Volume'].rolling(21).std()
        features['volume_breakout'] = (
            (data['Volume'] > data['Volume'].rolling(21).mean() + 2 * volume_std)
        ).astype(float)
        
        # Accumulation/Distribution
        features['ad_line'] = ta.ad(data['High'], data['Low'], data['Close'], data['Volume'])
        features['ad_trend'] = features['ad_line'] > features['ad_line'].shift(1)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        return [
            'volume_ma_5', 'volume_ma_10', 'volume_ma_21',
            'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_21',
            'obv', 'obv_ma', 'obv_trend',
            'volume_price_trend', 'vpt_ma',
            'volume_breakout',
            'ad_line', 'ad_trend'
        ]