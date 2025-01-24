from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
import logging
from dataclasses import dataclass
from enum import Enum

# Import feature group classes
from src.features.Features import PriceFeatures, ReturnFeatures, MomentumFeatures, VolatilityFeatures, VolumeFeatures

class ScalingMethod(Enum):
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"

@dataclass
class FeatureGroup:
    """Container for feature group configuration"""
    name: str
    scaler_type: ScalingMethod
    enabled: bool = True
    
class FeatureEngineering:
    """
    Coordinates feature generation and scaling across different feature groups.
    Enhanced to handle multi-index (datetime, symbol) data structure.
    """
    
    def __init__(self, config: Dict) -> None:
        """
        Initialize feature engineering with configuration.
        
        Args:
            config: Configuration dictionary containing feature parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize feature groups with appropriate scaling methods
        self.feature_groups = [
            FeatureGroup("price", ScalingMethod.ROBUST),
            FeatureGroup("returns", ScalingMethod.ROBUST),
            FeatureGroup("momentum", ScalingMethod.STANDARD),
            FeatureGroup("volatility", ScalingMethod.STANDARD),
            FeatureGroup("volume", ScalingMethod.ROBUST)
        ]
        
        # Initialize feature generators
        self.feature_generators = {
            "price": PriceFeatures(config.get("price_features", {})),
            "returns": ReturnFeatures(config.get("return_features", {})),
            "momentum": MomentumFeatures(config.get("momentum_features", {})),
            "volatility": VolatilityFeatures(config.get("volatility_features", {})),
            "volume": VolumeFeatures(config.get("volume_features", {}))
        }
        
        # Initialize scalers
        self._initialize_scalers()
        
        self.logger.info("FeatureEngineering initialized with multi-index support")

    def _initialize_scalers(self) -> None:
        """Initialize scaler dictionaries for each feature group and base columns."""
        self.scalers = {
            group.name: {} for group in self.feature_groups
        }
        # Add a new dictionary for base scalers (one per symbol)
        self.base_scalers = {}

    def _get_or_create_scaler(self, group_name: str, symbol: str) -> object:
        """Get or create a scaler for a specific group and symbol."""
        if symbol not in self.scalers[group_name]:
            group = next(g for g in self.feature_groups if g.name == group_name)
            if group.scaler_type == ScalingMethod.STANDARD:
                self.scalers[group_name][symbol] = StandardScaler()
            else:
                self.scalers[group_name][symbol] = RobustScaler()
        return self.scalers[group_name][symbol]
    
    def _validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate multi-index data structure and content.
        Returns tuple of (is_valid, excluded_symbols)
        """
        excluded_symbols = []
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        # Check required columns first
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Handle multi-index validation
        if isinstance(data.index, pd.MultiIndex):
            if len(data.index.levels) != 2:
                raise ValueError("Multi-index must have exactly 2 levels (datetime, symbol)")
            if not isinstance(data.index.levels[0], pd.DatetimeIndex):
                raise ValueError("First index level must be DatetimeIndex")
                
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = set(required_columns) - set(data.columns)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Check minimum samples per symbol
        if isinstance(data.index, pd.MultiIndex):
            min_samples = self.config.get("min_samples", 100)
            symbol_counts = data.groupby(level=1).size()
            insufficient_symbols = symbol_counts[symbol_counts < min_samples].index
            if len(insufficient_symbols) > 0:
                excluded_symbols.extend(insufficient_symbols)
                self.logger.warning(
                    f"Found symbols with insufficient data: {list(insufficient_symbols)}. "
                    f"Need at least {min_samples} samples per symbol. These will be excluded."
                )
        
        return len(excluded_symbols) < len(data.index.get_level_values(1).unique()), excluded_symbols

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features for the input data.
        
        Args:
            data: Input OHLCV DataFrame
            
        Returns:
            DataFrame containing all generated features
        """
        try:
            # Validate data and get excluded symbols
            is_valid, excluded_symbols = self._validate_data(data)
            
            if not isinstance(data.index, pd.MultiIndex):
                self.logger.warning("Input data is not multi-indexed. Converting to single symbol format.")
                return self._process_single_symbol(data)
            
            # Get valid symbols (excluding those with insufficient data)
            symbols = data.index.get_level_values(1).unique()
            valid_symbols = [sym for sym in symbols if sym not in excluded_symbols]
            
            if not valid_symbols:
                raise ValueError("No valid symbols remain after filtering")
            
            if excluded_symbols:
                self.logger.info(f"Processing {len(valid_symbols)} symbols, excluded {len(excluded_symbols)} symbols")
            
            processed_dfs = []
            for symbol in valid_symbols:
                self.logger.info(f"Processing features for symbol {symbol}")
                symbol_data = data.xs(symbol, level=1)
                processed_symbol = self._process_single_symbol(symbol_data)
                
                # Validate columns before adding to list
                expected_columns = self.get_all_expected_columns()
                if set(processed_symbol.columns) != set(expected_columns):
                    self.logger.warning(f"Column mismatch for {symbol}. Reindexing...")
                    processed_symbol = processed_symbol.reindex(columns=expected_columns)
                
                processed_symbol['symbol'] = symbol
                processed_symbol.set_index('symbol', append=True, inplace=True)
                processed_dfs.append(processed_symbol)
            
            if not processed_dfs:
                raise ValueError("No data processed successfully")
                
            all_features = pd.concat(processed_dfs)
            all_features.sort_index(inplace=True)
            return all_features
            
        except Exception as e:
            self.logger.error(f"Feature generation failed: {str(e)}")
            raise
    
    def _process_single_symbol(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process features for a single symbol with column validation."""
        try:
            # Get expected columns
            expected_columns = self.get_all_expected_columns()
            
            # Initialize with original OHLCV data
            all_features = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Generate features for each enabled group
            for group in self.feature_groups:
                if not group.enabled:
                    continue
                    
                self.logger.info(f"Generating {group.name} features")
                
                try:
                    generator = self.feature_generators[group.name]
                    features = generator.generate(data)
                    
                    # Validate feature generation output
                    if features is None:
                        self.logger.warning(f"{group.name} generator returned None. Skipping.")
                        continue
                        
                    if not isinstance(features, pd.DataFrame):
                        self.logger.warning(f"{group.name} generator returned non-DataFrame. Skipping.")
                        continue
                    
                    if features.empty:
                        self.logger.warning(f"{group.name} generator returned empty DataFrame. Skipping.")
                        continue
                    
                    # Add prefix to avoid name collisions
                    features = features.add_prefix(f"{group.name}_")
                    
                    # Replace any None values with NaN before concatenation
                    features = features.fillna(value=np.nan)
                    
                    # Validate feature columns match expected
                    expected_group_cols = [
                        col for col in expected_columns 
                        if col.startswith(f"{group.name}_")
                    ]
                    
                    if set(features.columns) != set(expected_group_cols):
                        missing = set(expected_group_cols) - set(features.columns)
                        extra = set(features.columns) - set(expected_group_cols)
                        
                        if missing:
                            self.logger.warning(f"Missing expected features: {missing}")
                            # Add missing columns as NaN
                            for col in missing:
                                features[col] = np.nan
                                
                        if extra:
                            self.logger.warning(f"Extra features found: {extra}")
                            # Remove extra columns
                            features = features[expected_group_cols]
                    
                    # Convert any remaining None values to NaN
                    features = features.replace({None: np.nan})
                    
                    # Add to results
                    all_features = pd.concat([all_features, features], axis=1)
                    
                except Exception as e:
                    self.logger.error(f"Error generating {group.name} features: {str(e)}")
                    # Instead of raising, we'll continue with other feature groups
                    continue
            
            # Ensure all expected columns are present in correct order
            all_features = all_features.reindex(columns=expected_columns)
            
            # Handle invalid values before returning
            try:
                return self._handle_invalid_values(all_features)
            except Exception as e:
                self.logger.error(f"Error handling invalid values: {str(e)}")
                # If invalid value handling fails, return the original features with NaN fill
                return all_features.fillna(0)
                
        except Exception as e:
            self.logger.error(f"Critical error in _process_single_symbol: {str(e)}")
            raise

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate and scale features while maintaining symbol separation.
        Fits separate scalers for each symbol and feature group.
        """
        features = self.process(data)
        
        if not isinstance(features.index, pd.MultiIndex):
            return self._fit_transform_single(features)
        
        # Process each symbol separately
        symbols = features.index.get_level_values(1).unique()
        scaled_dfs = []
        
        for symbol in symbols:
            # Get symbol data
            symbol_features = features.xs(symbol, level=1)
            
            # Scale features
            scaled_symbol = self._fit_transform_single(symbol_features, symbol)
            
            # Restore multi-index
            scaled_df = pd.DataFrame(
                scaled_symbol,
                index=symbol_features.index,
                columns=features.columns
            )
            scaled_df['symbol'] = symbol
            scaled_df.set_index('symbol', append=True, inplace=True)
            
            scaled_dfs.append(scaled_df)
        
        # Combine all scaled data
        scaled_features = pd.concat(scaled_dfs)
        scaled_features.sort_index(inplace=True)
        
        return scaled_features.values

    def _fit_transform_single(self, features: pd.DataFrame, symbol: str = None) -> np.ndarray:
        """
        Fit and transform features for a single symbol.
        Now stores the base scaler for later use.
        """
        # Initialize with all columns from input features
        scaled_features = pd.DataFrame(index=features.index, columns=features.columns)
        
        # Handle OHLCV columns with RobustScaler
        base_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        symbol_key = symbol or 'default'
        
        # Create and fit base scaler
        self.base_scalers[symbol_key] = RobustScaler()
        scaled_features[base_cols] = self.base_scalers[symbol_key].fit_transform(features[base_cols])
        
        # Handle generated features by group
        for group in self.feature_groups:
            if not group.enabled:
                continue
                
            # Get columns for this group
            group_cols = [col for col in features.columns 
                        if col.startswith(f"{group.name}_")]
            
            if group_cols:
                # Get or create scaler
                scaler = self._get_or_create_scaler(group.name, symbol_key)
                scaled = scaler.fit_transform(features[group_cols])
                scaled_features[group_cols] = scaled
        
        # Ensure we have all columns in the same order
        return scaled_features[features.columns].values

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scalers, maintaining symbol separation."""
        features = self.process(data)
        
        if not isinstance(features.index, pd.MultiIndex):
            return self._transform_single(features)
        
        # Process each symbol separately
        symbols = features.index.get_level_values(1).unique()
        scaled_dfs = []
        
        for symbol in symbols:
            # Get symbol data
            symbol_features = features.xs(symbol, level=1)
            
            # Scale features
            scaled_symbol = self._transform_single(symbol_features, symbol)
            
            # Restore multi-index
            scaled_df = pd.DataFrame(
                scaled_symbol,
                index=symbol_features.index,
                columns=features.columns
            )
            scaled_df['symbol'] = symbol
            scaled_df.set_index('symbol', append=True, inplace=True)
            
            scaled_dfs.append(scaled_df)
        
        # Combine all scaled data
        scaled_features = pd.concat(scaled_dfs)
        scaled_features.sort_index(inplace=True)
        
        return scaled_features.values

    def _transform_single(self, features: pd.DataFrame, symbol: str = None) -> np.ndarray:
        """
        Transform features for a single symbol using fitted scalers.
        Now uses the stored base scaler.
        """
        # Initialize with all columns from input features
        scaled_features = pd.DataFrame(index=features.index, columns=features.columns)
        
        # Handle OHLCV columns with stored RobustScaler
        base_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        symbol_key = symbol or 'default'
        
        # Use stored base scaler
        if symbol_key not in self.base_scalers:
            raise ValueError(f"No fitted base scaler found for symbol {symbol_key}. Must fit before transform.")
        scaled_features[base_cols] = self.base_scalers[symbol_key].transform(features[base_cols])
        
        # Handle generated features by group
        for group in self.feature_groups:
            if not group.enabled:
                continue
                
            # Get columns for this group
            group_cols = [col for col in features.columns 
                        if col.startswith(f"{group.name}_")]
            
            if group_cols:
                # Get scaler
                if symbol_key not in self.scalers[group.name]:
                    raise ValueError(f"No fitted scaler found for {group.name} features of symbol {symbol_key}")
                scaler = self.scalers[group.name][symbol_key]
                scaled = scaler.transform(features[group_cols])
                scaled_features[group_cols] = scaled
        
        # Ensure we have all columns in the same order
        return scaled_features[features.columns].values
    
    def _handle_invalid_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Handle invalid values in features DataFrame.
        
        Args:
            features: DataFrame of generated features
            
        Returns:
            DataFrame with invalid values handled
        """
        try:
            # Make a copy to avoid modifying the original
            df = features.copy()
            
            # Replace None values first
            df = df.replace({None: np.nan})
            
            # Convert all columns to numeric, coercing errors to NaN
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    self.logger.warning(f"Error converting column {col} to numeric: {str(e)}")
                    df[col] = df[col].replace([np.inf, -np.inf, None], np.nan)
            
            # Replace infinite values with NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill NaN values (use previous valid value)
            df = df.ffill()
            
            # Backward fill any remaining NaN values at the start
            df = df.bfill()
            
            # Fill any remaining NaN values with 0
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error handling invalid values: {str(e)}")
            # Return original with simple NaN filling as fallback
            return features.fillna(0)

    def get_all_expected_columns(self) -> List[str]:
        """Get complete list of expected columns including OHLCV"""
        # Start with base OHLCV columns
        columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Add prefixed feature columns from each group
        for group in self.feature_groups:
            if group.enabled:
                generator = self.feature_generators[group.name]
                group_features = [
                    f"{group.name}_{feature}"
                    for feature in generator.get_feature_names()
                ]
                columns.extend(group_features)
        
        return columns