import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import ta
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
from enum import Enum

from utils import validate_temporal_split, check_forward_looking_features

class ScalingMethod(Enum):
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"

@dataclass
class FeatureGroup:
    """Container for feature groups and their scaling methods"""
    name: str
    features: List[str]
    scaling_method: ScalingMethod

class FeatureEngineering:
    """Feature Engineering class for generating and transforming features"""
    
    def __init__(self, config: Dict):
        """Initialize feature engineering with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize scalers as dictionary of feature groups
        self.scalers = {}
        
        # Get parameters from config
        self.sequence_length = self.config['model'].get('sequence_length', 10)
        self.feature_config = self.config['features']
        
        # Initialize feature groups
        self._initialize_feature_groups()
        
    def _initialize_feature_groups(self):
        """Initialize feature groups and their respective scalers"""
        self.feature_groups = [
            FeatureGroup("price", [], ScalingMethod.ROBUST),
            FeatureGroup("returns", [], ScalingMethod.ROBUST),
            FeatureGroup("momentum", [], ScalingMethod.STANDARD),
            FeatureGroup("volatility", [], ScalingMethod.STANDARD),
            FeatureGroup("volume", [], ScalingMethod.ROBUST)
        ]
        
        for group in self.feature_groups:
            if group.scaling_method == ScalingMethod.STANDARD:
                self.scalers[group.name] = StandardScaler()
            else:
                self.scalers[group.name] = RobustScaler()

    def get_config(self) -> Dict:
        """Return current feature engineering configuration"""
        return self.config
        
    def update_config(self, config: Dict) -> None:
        """Update feature engineering configuration"""
        self.config = config

    def _generate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate price-based features with proper handling of infinities, zeros, and negative values.
        
        Args:
            df (pd.DataFrame): Input dataframe with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame containing price-based features
        """
        features = pd.DataFrame(index=df.index)
        
        # Safe log price calculation
        # Add small epsilon to avoid log(0) and handle negative prices
        epsilon = 1e-10
        safe_close = np.where(df['Close'] > 0, df['Close'], epsilon)
        features['log_price'] = np.log(safe_close)
        
        # Safe ratio calculations with handling for zero values
        features['high_low_ratio'] = np.where(
            df['Low'] > epsilon,
            df['High'] / df['Low'],
            1.0  # Default to 1.0 for invalid values
        )
        
        features['close_open_ratio'] = np.where(
            df['Open'] > epsilon,
            df['Close'] / df['Open'],
            1.0  # Default to 1.0 for invalid values
        )
        
        # Rolling price features with safety checks
        windows = [5, 10, 21]
        for window in windows:
            # Moving averages
            features[f'price_ma_{window}'] = df['Close'].rolling(
                window=window,
                min_periods=1
            ).mean()
            
            # Standard deviation with minimum value threshold
            std_values = df['Close'].rolling(
                window=window,
                min_periods=1
            ).std()
            features[f'price_std_{window}'] = np.where(
                std_values > epsilon,
                std_values,
                epsilon
            )
            
            # Skewness (already handles zero values appropriately)
            features[f'price_skew_{window}'] = df['Close'].rolling(
                window=window,
                min_periods=1
            ).skew()
        
        # Distance from support/resistance levels with safety checks
        for window in [10, 20]:
            rolling_high = df['High'].rolling(window, min_periods=1).max()
            rolling_low = df['Low'].rolling(window, min_periods=1).min()
            
            # Safe calculation of distance from high
            features[f'dist_high_{window}'] = np.where(
                rolling_high > epsilon,
                df['Close'] / rolling_high,
                1.0
            )
            
            # Safe calculation of distance from low
            features[f'dist_low_{window}'] = np.where(
                rolling_low > epsilon,
                df['Close'] / rolling_low,
                1.0
            )
        
        # Handle any remaining infinities or NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill NaN values, then backward fill any remaining
        # This ensures we don't have any NaN values in our features
        features = features.ffill().bfill()
        
        # Add final safety check to ensure no invalid values remain
        # Replace any remaining invalid values with 0
        features = features.fillna(0)
        
        return features

    def _generate_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate return-based features"""
        features = pd.DataFrame(index=df.index)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        close_prices = df['Close'].clip(lower=epsilon)
        
        # Log returns at different horizons
        horizons = [1, 3, 5, 10, 21]
        for horizon in horizons:
            features[f'log_return_{horizon}d'] = np.log(close_prices / close_prices.shift(horizon))
            
        # Rolling return statistics
        windows = [5, 10, 21]
        returns = np.log(close_prices / close_prices.shift(1))
        for window in windows:
            features[f'return_ma_{window}'] = returns.rolling(window).mean()
            features[f'return_std_{window}'] = returns.rolling(window).std()
            features[f'return_skew_{window}'] = returns.rolling(window).skew()
        
        # Handle any infinities or NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill()
        
        return features

    def _generate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum indicators"""
        features = pd.DataFrame(index=df.index)
        
        # RSI
        features['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        features['macd'] = macd.macd()
        features['macd_signal'] = macd.macd_signal()
        features['macd_diff'] = macd.macd_diff()
        
        # Rate of Change
        for period in [5, 10, 21]:
            features[f'roc_{period}'] = ta.momentum.ROCIndicator(df['Close'], period).roc()
        
        return features

    def _generate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility indicators"""
        features = pd.DataFrame(index=df.index)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        features['bb_high'] = bb.bollinger_hband()
        features['bb_low'] = bb.bollinger_lband()
        features['bb_width'] = (features['bb_high'] - features['bb_low']) / df['Close']
        
        # Average True Range
        features['atr'] = ta.volatility.AverageTrueRange(
            high=df['High'], 
            low=df['Low'], 
            close=df['Close']
        ).average_true_range()
        
        return features

    def _generate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based features"""
        features = pd.DataFrame(index=df.index)
        
        # Basic volume features
        features['log_volume'] = np.log(df['Volume'])
        
        # On-Balance Volume
        features['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['Close'],
            volume=df['Volume']
        ).on_balance_volume()
        
        # Volume moving averages
        windows = [5, 10, 21]
        for window in windows:
            features[f'volume_ma_{window}'] = df['Volume'].rolling(window).mean()
            features[f'volume_std_{window}'] = df['Volume'].rolling(window).std()
        
        # Volume-price relationship
        features['volume_price_corr'] = (
            df['Close'].rolling(10)
            .corr(df['Volume'])
        )
        
        return features

    def _generate_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate all features for all symbols"""
        print("\n=== Starting Feature Generation ===")
        all_features = []
        
        for symbol, df in data.items():
            print(f"\nProcessing symbol: {symbol}")
            # Generate all feature groups
            symbol_features = pd.DataFrame(index=df.index)
            
            feature_generators = {
                'price': self._generate_price_features,
                'returns': self._generate_return_features,
                'momentum': self._generate_momentum_features,
                'volatility': self._generate_volatility_features,
                'volume': self._generate_volume_features
            }
            
            # Generate features for each group
            for group_name, generator in feature_generators.items():
                print(f"Generating {group_name} features...")
                try:
                    group_features = generator(df)
                    print(f"{group_name} features shape: {group_features.shape}")
                    print(f"{group_name} features columns: {group_features.columns.tolist()}")
                    
                    # Add prefix to avoid name collisions
                    group_features = group_features.add_prefix(f'{group_name}_')
                    symbol_features = pd.concat([symbol_features, group_features], axis=1)
                except Exception as e:
                    print(f"Error generating {group_name} features: {str(e)}")
                    raise
            
            print(f"\nTotal features for {symbol}: {len(symbol_features.columns)}")
            print(f"Feature names: {symbol_features.columns.tolist()}")
            
            # Add symbol identifier
            symbol_features['symbol'] = symbol
            all_features.append(symbol_features)
            
            # Only print for first symbol to avoid cluttering output
            break
        
        # Combine all symbols
        print("\nCombining features for all symbols...")
        combined_features = pd.concat(all_features)
        
        # Drop any remaining NaN values from feature calculation
        print(f"\nShape before dropping NaN: {combined_features.shape}")
        combined_features = combined_features.dropna()
        print(f"Shape after dropping NaN: {combined_features.shape}")
        
        return combined_features

    def _scale_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Scale features by group with infinity handling"""
        print("\n=== Starting Feature Scaling ===")
        print(f"Input features shape: {features.shape}")
        print(f"Input feature columns: {features.columns.tolist()}")
        
        scaled_features = pd.DataFrame(index=features.index)
        
        # Replace any remaining infinities with NaN
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill NaN values, then backward fill any remaining
        features = features.ffill().bfill()
        
        for group in self.feature_groups:
            group_cols = [col for col in features.columns if col.startswith(f'{group.name}_')]
            print(f"\nScaling {group.name} features...")
            print(f"Found {len(group_cols)} columns for {group.name}")
            print(f"Columns: {group_cols}")
            
            if group_cols:
                try:
                    scaled = self.scalers[group.name].fit_transform(features[group_cols])
                    scaled_features[group_cols] = scaled
                    print(f"Successfully scaled {len(group_cols)} {group.name} features")
                except Exception as e:
                    print(f"Error scaling {group.name} features: {str(e)}")
                    raise
        
        # Add symbol column back if it exists
        if 'symbol' in features.columns:
            scaled_features['symbol'] = features['symbol']
        
        print(f"\nFinal scaled features shape: {scaled_features.shape}")
        print(f"Final scaled feature columns: {scaled_features.columns.tolist()}")
        
        return scaled_features.values

    def _generate_labels(self, data: Dict[str, pd.DataFrame]) -> pd.Series:
        """Generate labels for training"""
        all_labels = []
        
        for symbol, df in data.items():
            # Calculate forward returns
            horizon = self.config['features'].get('target_horizon', 1)
            returns = df['Close'].pct_change(horizon).shift(-horizon)
            all_labels.append(returns)
        
        labels = pd.concat(all_labels)
        labels = labels[~labels.isna()]  # Remove NaN values
        
        return labels

    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """Create sequences for LSTM input"""
        print("\n=== Creating Sequences ===")
        print(f"Input data shape: {data.shape}")
        
        # If data is a DataFrame, convert to numpy array
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Remove the 'symbol' column if it exists (it should be the last column)
        if data.shape[1] > 0:  # Check if we have any columns
            data = data[:, :-1]  # Remove last column (symbol)
        
        print(f"Data shape after removing symbol column: {data.shape}")
        
        # Check if we have enough data
        if len(data) < sequence_length:
            raise ValueError(f"Not enough data points ({len(data)}) for sequence length {sequence_length}")
        
        # Create sequences
        n_samples = len(data) - sequence_length + 1
        n_features = data.shape[1]
        
        print(f"Creating sequences with:")
        print(f"n_samples: {n_samples}")
        print(f"sequence_length: {sequence_length}")
        print(f"n_features: {n_features}")
        
        # Additional validation
        if n_samples <= 0:
            raise ValueError(f"Invalid n_samples: {n_samples}. Data length: {len(data)}, sequence_length: {sequence_length}")
        if n_features <= 0:
            raise ValueError(f"Invalid n_features: {n_features}")
        
        # Initialize output array
        sequences = np.zeros((n_samples, sequence_length, n_features))
        
        # Fill sequences
        for i in range(n_samples):
            sequences[i] = data[i:i + sequence_length]
        
        print(f"Final sequences shape: {sequences.shape}")
        return sequences

    def transform(self, historical_data: Dict[str, pd.DataFrame], train_size: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform raw data into features and split into train/test sets"""
        print("\n=== Starting Feature Generation with Leakage Prevention ===")
        
        # Sort data by time to ensure temporal order
        sorted_data = {
            symbol: df.sort_index() for symbol, df in historical_data.items()
        }
        
        # Calculate split point
        sample_symbol = list(sorted_data.keys())[0]
        total_dates = len(sorted_data[sample_symbol])
        split_idx = int(total_dates * train_size)
        split_date = sorted_data[sample_symbol].index[split_idx]
        
        print(f"\nTemporal split date: {split_date}")
        
        # Split data before feature generation to prevent leakage
        train_data = {
            symbol: df.loc[:split_date] 
            for symbol, df in sorted_data.items()
        }
        test_data = {
            symbol: df.loc[split_date:] 
            for symbol, df in sorted_data.items()
        }
        
        # Generate and scale features separately for train and test
        print("\n=== Processing Training Data ===")
        train_features = self._generate_features(train_data)
        train_labels = self._generate_labels(train_data)
        
        # Check for forward-looking features in training data
        if self.config['validation'].get('feature_checks', {}).get('enabled', True):
            print(check_forward_looking_features(train_features))
        
        # Fit scalers on training data only
        self._fit_scalers(train_features)
        
        # Scale training data
        scaled_train_features = self._transform_features(train_features)
        
        print("\n=== Processing Test Data ===")
        test_features = self._generate_features(test_data)
        test_labels = self._generate_labels(test_data)
        
        # Scale test data using scalers fit on training data
        scaled_test_features = self._transform_features(test_features)
        
        # Create sequences
        X_train = self._create_sequences(scaled_train_features, self.sequence_length)
        X_test = self._create_sequences(scaled_test_features, self.sequence_length)
        
        # Adjust labels
        y_train = train_labels[self.sequence_length-1:].values
        y_test = test_labels[self.sequence_length-1:].values
        
        # Verify temporal validity
        train_dates = train_features.index
        test_dates = test_features.index
        if not validate_temporal_split(train_dates, test_dates):
            raise ValueError("Temporal split validation failed - potential data leakage detected")
        
        print("\n=== Data Split Statistics ===")
        print(f"Training dates: {min(train_dates)} to {max(train_dates)}")
        print(f"Testing dates: {min(test_dates)} to {max(test_dates)}")
        
        return X_train, X_test, y_train, y_test

    def _fit_scalers(self, features: pd.DataFrame) -> None:
        """Fit scalers on training data only"""
        for group in self.feature_groups:
            group_cols = [col for col in features.columns if col.startswith(f'{group.name}_')]
            if group_cols:
                self.scalers[group.name].fit(features[group_cols])

    def _transform_features(self, features: pd.DataFrame) -> np.ndarray:
        """Transform features using pre-fit scalers"""
        scaled_features = pd.DataFrame(index=features.index)
        
        for group in self.feature_groups:
            group_cols = [col for col in features.columns if col.startswith(f'{group.name}_')]
            if group_cols:
                scaled = self.scalers[group.name].transform(features[group_cols])
                scaled_features[group_cols] = scaled
        
        return scaled_features.values

    def transform_predict(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Transform data for prediction"""
        print("\n=== Starting prediction transformation ===")
        
        try:
            # Generate features
            print("Generating features...")
            features = self._generate_features(data)
            print(f"Generated features shape: {features.shape}")
            
            # Scale features using fitted scalers
            print("Scaling features...")
            scaled_features = self._scale_features(features)
            print(f"Scaled features shape: {scaled_features.shape}")
            
            # Create sequences by symbol
            sequences = {}
            for symbol in data.keys():
                print(f"\nProcessing symbol: {symbol}")
                
                # Get symbol data
                symbol_mask = features['symbol'] == symbol
                symbol_data = scaled_features[symbol_mask]
                
                print(f"Symbol data shape before sequence creation: {symbol_data.shape}")
                
                # Ensure we have enough data
                if len(symbol_data) < self.sequence_length:
                    print(f"Warning: Not enough data for {symbol} to create sequences")
                    continue
                    
                try:
                    sequences[symbol] = self._create_sequences(
                        symbol_data,
                        self.sequence_length
                    )
                    print(f"Created sequences for {symbol} with shape: {sequences[symbol].shape}")
                except Exception as e:
                    print(f"Error creating sequences for {symbol}: {str(e)}")
                    continue
            
            if not sequences:
                raise ValueError("No valid sequences could be created for any symbol")
                
            return sequences
            
        except Exception as e:
            print(f"Error in transform_predict: {str(e)}")
            raise