from typing import Dict, Any
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import torch
import yaml
import logging
from pathlib import Path
import numpy as np
from typing import Tuple
from LSTM import DirectionalLSTM
from src.features.FeatureEngineer import FeatureEngineering
from LSTM import LSTMConfig
from src.models.HyperparameterOptimizer import HyperparameterOptimizer

CONFIG_PATH = Path("config/config.yaml")

class TimeSeriesSplitter:
    """
    Handles time series data splitting with proper temporal ordering and gap handling.
    Modified to support multi-index DataFrame for multiple stocks.
    """
    def __init__(self, validation_ratio: float = 0.2, gap_days: int = 5):
        self.validation_ratio = validation_ratio
        self.gap_days = gap_days
        
    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split time series data while maintaining temporal order and adding a gap.
        Handles multi-index DataFrame with (date, symbol) hierarchy.
        
        Args:
            data: DataFrame with MultiIndex (DatetimeIndex, symbol)
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and validation splits
        """
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Data must have MultiIndex (datetime, symbol)")
            
        # Get unique symbols
        symbols = data.index.get_level_values(1).unique()
        
        train_dfs = []
        val_dfs = []
        
        for symbol in symbols:
            # Get data for this symbol
            symbol_data = data.xs(symbol, level=1)
            
            # Calculate split point
            split_idx = int(len(symbol_data) * (1 - self.validation_ratio))
            split_date = symbol_data.index[split_idx]
            
            # Create gap
            gap_start = split_date
            gap_end = split_date + pd.Timedelta(days=self.gap_days)
            
            # Split data
            train_data = symbol_data[:gap_start]
            val_data = symbol_data[gap_end:]
            
            # Restore multi-index
            train_data = train_data.assign(symbol=symbol)
            val_data = val_data.assign(symbol=symbol)
            
            train_data.set_index('symbol', append=True, inplace=True)
            val_data.set_index('symbol', append=True, inplace=True)
            
            train_dfs.append(train_data)
            val_dfs.append(val_data)
        
        # Combine all splits while maintaining multi-index
        train_combined = pd.concat(train_dfs)
        val_combined = pd.concat(val_dfs)
        
        # Sort index
        train_combined.sort_index(inplace=True)
        val_combined.sort_index(inplace=True)
        
        return train_combined, val_combined

class LSTMManager:
    """
    Manages a directional prediction LSTM model for financial time series data.
    Handles model training, prediction, and feature engineering.
    
    The class is configured via a config.yaml file and maintains minimal state.
    """
    
    def __init__(self) -> None:
        """
        Initialize the LSTM manager by loading configuration and setting up components.
        
        Raises:
            FileNotFoundError: If config.yaml is not found
            yaml.YAMLError: If config.yaml is invalid
            ValueError: If required config parameters are missing
        """
        self._setup_logging()
        self.config = self._load_config()
        self._validate_config()
        
        self.feature_engineering = FeatureEngineering(self.config['feature_params'])
        
        lstm_config = LSTMConfig(
            input_size=self.config['model_params']['input_size'],
            hidden_size=self.config['model_params']['hidden_size'],
            num_layers=self.config['model_params']['num_layers'],
            dropout=self.config['model_params']['dropout'],
            bidirectional=self.config['model_params']['bidirectional'],
            attention_heads=self.config['model_params']['attention_heads'],
            use_layer_norm=self.config['model_params']['use_layer_norm'],
            residual_connections=self.config['model_params']['residual_connections'],
            confidence_threshold=self.config['model_params']['confidence_threshold']
        )
        
        self.model = DirectionalLSTM(lstm_config)
        logging.info("LSTMManager initialized with multi-index support")
    
    def _setup_logging(self) -> None:
        """Configure logging with appropriate format and level."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load and parse the config.yaml file.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
            
        Raises:
            FileNotFoundError: If config.yaml is not found
            yaml.YAMLError: If config.yaml is invalid
        """
        try:
            with open(CONFIG_PATH) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.error(f"Configuration file not found at {CONFIG_PATH}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing configuration file: {e}")
            raise
    
    def _validate_config(self) -> None:
        """
        Validate that all required configuration parameters are present.
        
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        required_sections = ['model_params', 'feature_params', 'training_params', 
                           'early_stopping_params', 'prediction_params']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate specific required parameters
        if 'sequence_length' not in self.config['model_params']:
            raise ValueError("Missing required parameter: model_params.sequence_length")
        
        if 'confidence_threshold' not in self.config['prediction_params']:
            raise ValueError("Missing required parameter: prediction_params.confidence_threshold")
    
    def _create_multi_index_data(self, historical_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Convert dictionary of DataFrames to multi-index DataFrame.
        
        Args:
            historical_data: Dictionary mapping symbols to DataFrames
            
        Returns:
            pd.DataFrame: Multi-index DataFrame with (date, symbol) hierarchy
        """
        dfs = []
        for symbol, df in historical_data.items():
            # Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Add symbol level
            df = df.assign(symbol=symbol)
            df.set_index('symbol', append=True, inplace=True)
            dfs.append(df)
        
        # Combine all DataFrames
        combined = pd.concat(dfs)
        
        # Sort index properly
        combined.sort_index(inplace=True)
        return combined

    def train(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Enhanced training with proper multi-index data handling and debug prints"""
        try:
            logging.info("Starting training with multi-index support...")
            print(f"\n[DEBUG] Initial historical data symbols: {list(historical_data.keys())}")
            print(f"[DEBUG] Sample data shape for first symbol: {next(iter(historical_data.values())).shape}")
            
            # Validate and filter historical data
            valid_data = self._validate_historical_data(historical_data)
            print(f"\n[DEBUG] After validation:")
            print(f"- Valid symbols remaining: {list(valid_data.keys())}")
            print(f"- Rows per symbol: {[df.shape[0] for df in valid_data.values()]}")
            
            # Convert to multi-index structure
            multi_index_data = self._create_multi_index_data(valid_data)
            print(f"\n[DEBUG] Multi-index data structure:")
            print(f"- Total shape: {multi_index_data.shape}")
            print(f"- Index levels: {multi_index_data.index.names}")
            print(f"- Number of unique symbols: {len(multi_index_data.index.get_level_values(1).unique())}")
            
            # Create splitter
            splitter = TimeSeriesSplitter(
                validation_ratio=self.config['training_params'].get('validation_ratio', 0.2),
                gap_days=self.config['training_params'].get('gap_days', 5)
            )
            print(f"\n[DEBUG] TimeSeriesSplitter configuration:")
            print(f"- Validation ratio: {splitter.validation_ratio}")
            print(f"- Gap days: {splitter.gap_days}")
            
            # Process features maintaining multi-index
            print("\n[DEBUG] Starting feature processing...")
            features = self.feature_engineering.process(multi_index_data)
            print(f"- Processed features shape: {features.shape}")
            print(f"- Feature columns: {list(features.columns)}")
            
            # Split features into train/val while maintaining symbol separation
            train_features, val_features = splitter.split(features)
            print(f"\n[DEBUG] After train/val split:")
            print(f"- Train set shape: {train_features.shape}")
            print(f"- Validation set shape: {val_features.shape}")
            print(f"- Train symbols: {len(train_features.index.get_level_values(1).unique())}")
            print(f"- Val symbols: {len(val_features.index.get_level_values(1).unique())}")
            
            # Scale features
            print("\n[DEBUG] Starting feature scaling...")
            scaled_train = self.feature_engineering.fit_transform(train_features)
            scaled_val = self.feature_engineering.transform(val_features)
            print(f"- Scaled train shape: {scaled_train.shape}")
            print(f"- Scaled val shape: {scaled_val.shape}")
            print(f"- Sample scaled values range: [{scaled_train.min():.3f}, {scaled_train.max():.3f}]")
            
            # Create sequences by symbol
            print("\n[DEBUG] Creating sequences...")
            train_sequences, train_labels = self._create_multi_symbol_sequences(
                scaled_train, train_features.index
            )
            val_sequences, val_labels = self._create_multi_symbol_sequences(
                scaled_val, val_features.index
            )
            print(f"- Train sequences shape: {train_sequences.shape}")
            print(f"- Train labels shape: {train_labels.shape}")
            print(f"- Val sequences shape: {val_sequences.shape}")
            print(f"- Val labels shape: {val_labels.shape}")
            print(f"- Label distribution - Train: {np.bincount(train_labels)}")
            print(f"- Label distribution - Val: {np.bincount(val_labels)}")
            
            # Validate sequence data
            self._validate_sequence_data(train_sequences, train_labels)
            self._validate_sequence_data(val_sequences, val_labels)
            print("\n[DEBUG] Sequence validation passed")
            
            # Compute class weights
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(train_labels),
                y=train_labels
            )
            self.config['training_params']['class_weights'] = dict(
                enumerate(class_weights)
            )
            print(f"\n[DEBUG] Computed class weights: {dict(enumerate(class_weights))}")
            
            # Convert to PyTorch format
            train_data = self._prepare_torch_data(train_sequences, train_labels)
            val_data = self._prepare_torch_data(val_sequences, val_labels)
            print(f"\n[DEBUG] DataLoader details:")
            print(f"- Train batches: {len(train_data)}")
            print(f"- Val batches: {len(val_data)}")
            print(f"- Batch size: {self.config['training_params'].get('batch_size', 32)}")
            
            # Train model
            print("\n[DEBUG] Starting model training...")
            metrics = self.model.train_model(
                train_loader=train_data,
                validation_loader=val_data,
                epochs=self.config['training_params'].get('epochs', 100),
                learning_rate=self.config['training_params'].get('learning_rate', 1e-3),
                weight_decay=self.config['training_params'].get('weight_decay', 1e-5),
                early_stopping_params=self.config['early_stopping_params'],
                class_weights=self.config['training_params'].get('class_weights'),
            )
            print(f"\n[DEBUG] Training completed. Final metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            print(f"\n[DEBUG] ERROR in training: {str(e)}")
            print(f"[DEBUG] Error type: {type(e).__name__}")
            print(f"[DEBUG] Stack trace:")
            import traceback
            traceback.print_exc()
            raise

    def _create_multi_symbol_sequences(
        self, 
        data: np.ndarray, 
        index: pd.MultiIndex
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences maintaining symbol separation.
        
        Args:
            data: Scaled feature data
            index: Multi-index containing (date, symbol)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Sequences and labels
        """
        sequence_length = self.config['model_params']['sequence_length']
        prediction_horizon = self.config['model_params']['prediction_horizon']
        threshold = self.config['model_params'].get('movement_threshold', 0.0)
        
        sequences = []
        labels = []
        
        # Get unique symbols
        symbols = index.get_level_values(1).unique()
        
        for symbol in symbols:
            # Get data for this symbol
            symbol_mask = index.get_level_values(1) == symbol
            symbol_data = data[symbol_mask]
            symbol_dates = index.get_level_values(0)[symbol_mask]
            
            for i in range(len(symbol_data) - sequence_length - prediction_horizon):
                # Get sequence
                seq = symbol_data[i:i+sequence_length]
                
                # Validate sequence
                if np.isnan(seq).any():
                    continue
                
                # Calculate future return
                current_price = symbol_data[i+sequence_length-1][3]  # Assuming Close is at index 3
                future_price = symbol_data[i+sequence_length+prediction_horizon-1][3]
                
                # Handle divide by zero case
                if current_price == 0 or np.isnan(current_price) or np.isnan(future_price):
                    continue
                    
                future_return = (future_price - current_price) / current_price
                
                # Skip if return is invalid
                if not np.isfinite(future_return):
                    continue
                
                # Apply threshold
                if abs(future_return) < threshold:
                    continue
                    
                label = int(future_return > 0)
                
                sequences.append(seq)
                labels.append(label)
        
        if not sequences:
            raise ValueError("No valid sequences created")
        
        # Convert to float32 before returning
        return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)

    def predict(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Generate prediction for a single symbol.
        
        Args:
            data: DataFrame for a single symbol
            symbol: Stock symbol
            
        Returns:
            Dictionary containing prediction details
        """
        try:
            logging.info(f"Starting prediction for symbol {symbol}")
            
            # Validate input data
            self._validate_prediction_data(data)
            
            # Add symbol level to index
            data = data.assign(symbol=symbol)
            data.set_index('symbol', append=True, inplace=True)
            
            # Process features
            features = self.feature_engineering.process(data)
            
            # Scale features
            scaled_features = self.feature_engineering.transform(features)
            
            # Create sequence
            sequence = self._create_prediction_sequence(scaled_features)
            self._validate_prediction_sequence(sequence)
            
            # Get prediction
            sequence_tensor = torch.FloatTensor(sequence).to(self.model.device)
            with torch.no_grad():
                raw_probability = self.model(sequence_tensor).sigmoid().cpu().numpy()[0]
            
            # Detect market regime and adjust probability
            market_regime = self._detect_market_regime(data)
            volatility = self._calculate_volatility(data)
            
            adjusted_probability = self._adjust_probability_for_regime(
                raw_probability,
                market_regime,
                volatility
            )
            
            # Create prediction
            if adjusted_probability < 0.5:
                direction = 0
                final_probability = 1 - adjusted_probability
            else:
                direction = 1
                final_probability = adjusted_probability
            
            prediction = {
                'symbol': symbol,
                'direction': direction,
                'probability': float(final_probability),
                'timestamp': data.index.get_level_values(0)[-1],
                'market_regime': market_regime,
                'volatility': float(volatility),
                'metadata': {
                    'raw_probability': float(raw_probability),
                    'sequence_length': self.config['model_params']['sequence_length'],
                    'prediction_horizon': self.config['model_params']['prediction_horizon']
                }
            }
            
            # Apply confidence threshold
            if abs(adjusted_probability - 0.5) < self.config['prediction_params']['confidence_threshold']:
                prediction['direction'] = None
            
            return prediction
            
        except Exception as e:
            logging.error(f"Prediction failed for {symbol}: {str(e)}")
            raise

    def _validate_historical_data(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Validate historical data and return filtered dictionary with valid symbols only.
        
        Args:
            historical_data: Dictionary mapping symbols to DataFrames
            
        Returns:
            Dict[str, pd.DataFrame]: Filtered dictionary with valid symbols only
        """
        if not historical_data:
            raise ValueError("Empty historical data provided")
            
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                        'Dividends', 'Stock Splits']
        
        valid_data = {}
        excluded_symbols = []
        
        for symbol, data in historical_data.items():
            try:
                # Basic DataFrame validation
                if not isinstance(data, pd.DataFrame):
                    logging.warning(f"Skipping {symbol}: Data must be a DataFrame")
                    excluded_symbols.append(symbol)
                    continue
                    
                # Check required columns
                missing_columns = set(required_columns) - set(data.columns)
                if missing_columns:
                    logging.warning(f"Skipping {symbol}: Missing columns {missing_columns}")
                    excluded_symbols.append(symbol)
                    continue
                    
                # Check minimum samples
                if len(data) < self.config['model_params']['sequence_length']:
                    logging.warning(
                        f"Skipping {symbol}: Insufficient data. Has {len(data)} rows, "
                        f"needs {self.config['model_params']['sequence_length']}"
                    )
                    excluded_symbols.append(symbol)
                    continue
                    
                # Ensure DatetimeIndex
                if not isinstance(data.index, pd.DatetimeIndex):
                    try:
                        data.index = pd.to_datetime(data.index)
                    except Exception as e:
                        logging.warning(f"Skipping {symbol}: Invalid index - {str(e)}")
                        excluded_symbols.append(symbol)
                        continue
                
                # If all checks pass, add to valid data
                valid_data[symbol] = data
                
            except Exception as e:
                logging.warning(f"Skipping {symbol} due to error: {str(e)}")
                excluded_symbols.append(symbol)
        
        if not valid_data:
            raise ValueError("No valid symbols remain after filtering")
        
        if excluded_symbols:
            logging.info(f"Excluded {len(excluded_symbols)} symbols: {excluded_symbols}")
            logging.info(f"Proceeding with {len(valid_data)} valid symbols")
        
        return valid_data

    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        Detect the current market regime based on recent price action.
        
        Args:
            data: DataFrame of OHLCV data
            
        Returns:
            str: One of ['trending_up', 'trending_down', 'ranging', 'high_volatility']
        """
        logging.info("Detecting market regime...")
        
        # Calculate required metrics
        returns = data['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Calculate trend metrics
        sma_20 = data['Close'].rolling(20).mean()
        sma_50 = data['Close'].rolling(50).mean()
        
        current_price = data['Close'].iloc[-1]
        current_sma20 = sma_20.iloc[-1]
        current_sma50 = sma_50.iloc[-1]
        
        # Define volatility threshold from config
        vol_threshold = self.config['prediction_params'].get('volatility_threshold', 0.25)
        
        logging.info(f"Current volatility: {volatility:.4f}, Threshold: {vol_threshold}")
        logging.info(f"Price: {current_price:.4f}, SMA20: {current_sma20:.4f}, SMA50: {current_sma50:.4f}")
        
        # Determine regime
        if volatility > vol_threshold:
            regime = 'high_volatility'
        elif current_price > current_sma20 > current_sma50:
            regime = 'trending_up'
        elif current_price < current_sma20 < current_sma50:
            regime = 'trending_down'
        else:
            regime = 'ranging'
            
        logging.info(f"Detected regime: {regime}")
        return regime

    def _calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        """
        Calculate recent volatility.
        
        Args:
            data: DataFrame of OHLCV data
            window: Rolling window for volatility calculation
            
        Returns:
            float: Annualized volatility
        """
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window).std().iloc[-1] * np.sqrt(252)
        logging.info(f"Calculated {window}-day annualized volatility: {volatility:.4f}")
        return volatility

    def _adjust_probability_for_regime(
        self,
        base_probability: float,
        market_regime: str,
        volatility: float
    ) -> float:
        """
        Adjust prediction probability based on market regime and volatility.
        
        Args:
            base_probability: Raw model probability
            market_regime: Detected market regime
            volatility: Current volatility level
            
        Returns:
            float: Adjusted probability
        """
        logging.info("Adjusting probability for market regime...")
        logging.info(f"Base probability: {base_probability:.4f}")
        
        # Get adjustment factors from config
        regime_factors = self.config['prediction_params'].get('regime_factors', {
            'trending_up': 1.1,
            'trending_down': 1.1,
            'ranging': 0.9,
            'high_volatility': 0.8
        })
        
        # Get volatility adjustment threshold
        vol_threshold = self.config['prediction_params'].get('volatility_threshold', 0.25)
        
        # Apply regime adjustment
        regime_factor = regime_factors.get(market_regime, 1.0)
        adjusted_probability = base_probability * regime_factor
        
        logging.info(f"After regime adjustment ({regime_factor}): {adjusted_probability:.4f}")
        
        # Apply volatility dampening if above threshold
        if volatility > vol_threshold:
            vol_dampening = 1.0 - min((volatility - vol_threshold) / vol_threshold, 0.5)
            adjusted_probability = 0.5 + (adjusted_probability - 0.5) * vol_dampening
            logging.info(f"After volatility dampening ({vol_dampening}): {adjusted_probability:.4f}")
        
        # Ensure probability stays in [0,1]
        adjusted_probability = min(max(adjusted_probability, 0.0), 1.0)
        
        logging.info(f"Final adjusted probability: {adjusted_probability:.4f}")
        return adjusted_probability
    
    def _validate_prediction_data(self, data: pd.DataFrame) -> None:
        """Validate prediction data format and content."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Prediction data must be a DataFrame")
            
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                          'Dividends', 'Stock Splits']
        
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        if len(data) < self.config['model_params']['sequence_length']:
            raise ValueError(
                f"Insufficient data. Need at least "
                f"{self.config['model_params']['sequence_length']} rows"
            )
    
    def _create_prediction_sequence(self, features: pd.DataFrame) -> np.ndarray:
        """Create sequence for prediction."""
        sequence_length = self.config['model_params']['sequence_length']
        return features.iloc[-sequence_length:].values.reshape(1, sequence_length, -1)
    
    def _prepare_torch_data(self, sequences: np.ndarray, labels: np.ndarray) -> torch.utils.data.DataLoader:
        """Prepare data for PyTorch training with explicit float32 dtype"""
        # Ensure float32 and int64 dtypes
        sequences = torch.tensor(sequences, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        dataset = torch.utils.data.TensorDataset(sequences, labels)
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['training_params'].get('batch_size', 32),
            shuffle=True
        )

    def _validate_sequence_data(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        """Validate sequence data before training"""
        if len(sequences) != len(labels):
            raise ValueError("Sequence and label lengths do not match")
        
        if sequences.shape[1] != self.config['model_params']['sequence_length']:
            raise ValueError("Sequences have incorrect sequence length")
        
        if not np.isfinite(sequences).all():
            raise ValueError("Sequences contain non-finite values")
        
    def _validate_prediction_sequence(self, sequence: np.ndarray) -> None:
        """Validate a single prediction sequence"""
        if sequence.shape[0] != 1:
            raise ValueError("Prediction sequence should have batch size 1")
            
        if sequence.shape[1] != self.config['model_params']['sequence_length']:
            raise ValueError("Sequence has incorrect sequence length")
            
        if not np.isfinite(sequence).all():
            raise ValueError("Sequence contains non-finite values")
        
    def optimize_hyperparameters(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run the complete hyperparameter optimization process using the HyperparameterOptimizer.
        
        Args:
            historical_data: Dictionary mapping symbols to DataFrames with historical data
            
        Returns:
            Dictionary containing:
            - best_parameters: The optimal parameters found
            - optimization_report: Path to the generated report
            - optimization_metrics: Dictionary of performance metrics
        """
        try:
            self.logger.info("Starting hyperparameter optimization process")
            
            # Get optimization config from main config
            optimization_config = self.config.get('optimization_params', {})
            if not optimization_config:
                raise ValueError("No optimization configuration found in config.yaml")
                
            # Initialize optimizer
            optimizer = HyperparameterOptimizer(self, optimization_config)
            
            # Convert data to multi-index DataFrame
            data = self._create_multi_index_data(historical_data)
            
            # Initialize optimization workflow
            search_space = optimizer.setup_search_space()
            optimizer.setup_validation_strategy(data)
            
            # Phase 1: Coarse Search
            self.logger.info("Starting coarse parameter search")
            coarse_params = optimizer.run_coarse_search(data)
            self.logger.info(f"Coarse search completed with parameters: {coarse_params}")
            
            # Phase 2: Feature Optimization
            self.logger.info("Starting feature optimization")
            feature_params = optimizer.optimize_features(data, coarse_params)
            self.logger.info(f"Feature optimization completed with parameters: {feature_params}")
            
            # Phase 3: Fine Tuning
            self.logger.info("Starting fine-tuning phase")
            final_params = optimizer.fine_tune_parameters(data, feature_params)
            self.logger.info(f"Fine-tuning completed with parameters: {final_params}")
            
            # Phase 4: Robustness Testing
            self.logger.info("Starting robustness testing")
            robustness_results = optimizer.run_robustness_tests(data)
            self.logger.info("Robustness testing completed")
            
            # Generate final report
            report_path = optimizer.generate_optimization_report()
            self.logger.info(f"Generated optimization report at: {report_path}")
            
            # Return comprehensive results
            results = {
                'best_parameters': final_params,
                'optimization_report': report_path,
                'robustness_metrics': robustness_results
            }
            
            self.logger.info("Hyperparameter optimization process completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {str(e)}")
            raise