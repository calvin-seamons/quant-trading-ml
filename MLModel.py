import os
import json
import logging
import traceback
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
from FeatureEngineering import FeatureEngineering
from HyperparameterOptimizer import HyperparameterOptimizer

from utils import calculate_financial_metrics
from LSTM import LSTMConfig, ImprovedLSTM

class TimeSeriesDataset(Dataset):
    """Custom Dataset for LSTM input"""
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int):
        """
        Initialize the dataset with proper reshaping
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, sequence_length, n_features)
            y (np.ndarray): Target values
            sequence_length (int): Length of each sequence
        """
        # Ensure X is 3D
        if X.ndim != 3:
            raise ValueError(f"Expected X to be 3D, got {X.ndim}D instead")
            
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single item from the dataset
        
        Args:
            idx (int): Index of the item
            
        Returns:
            tuple: (sequence, target) where sequence is of shape (sequence_length, n_features)
        """
        return self.X[idx], self.y[idx]

class MLModel:
    """
    Machine Learning Model Manager
    Handles model training, prediction, and persistence
    """
    def __init__(self, config: Dict):
        """
        Initialize MLModel with configuration
        
        Args:
            config (Dict): Configuration dictionary containing model parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model = None
        self.feature_engineering = FeatureEngineering(config)
        self.scaler = None
            
        # Get model parameters for filename
        model_version = config['model']['version']
        n_layers = config['model'].get('num_layers', 2)
        hidden_size = config['model'].get('hidden_size', 128)
        sequence_length = config['model'].get('sequence_length', 10)
        
        # Additional parameters for improved LSTM
        attention_heads = config['model'].get('attention_heads', 4)
        bidirectional = config['model'].get('bidirectional', True)
        
        # Create descriptive filename including new parameters
        model_name = (
            f"lstm_layers{n_layers}_hidden{hidden_size}_"
            f"seq{sequence_length}_attn{attention_heads}_"
            f"{'bi' if bidirectional else 'uni'}_v{model_version}"
        )
        
        # Setup paths
        self.model_dir = config['paths']['model_dir']
        self.model_path = os.path.join(
            self.model_dir,
            f"{model_name}.pt"
        )
        self.metadata_path = os.path.join(
            self.model_dir,
            f"{model_name}_metadata.json"
        )
        self.scaler_path = os.path.join(
            self.model_dir,
            f"{model_name}_scaler.joblib"
        )
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Model parameters
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.batch_size = config['model'].get('batch_size', 32)
        self.num_epochs = config['model'].get('num_epochs', 100)
        self.learning_rate = config['model'].get('learning_rate', 0.001)
        
        # Try to load existing model with fingerprint validation
        self._load_model()

    def _initialize_model(self) -> None:
        """Initialize the improved LSTM model"""
        # Create config for improved LSTM
        model_config = LSTMConfig(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.config['model'].get('dropout', 0.2),
            bidirectional=self.config['model'].get('bidirectional', True),
            attention_heads=self.config['model'].get('attention_heads', 4),
            use_layer_norm=self.config['model'].get('use_layer_norm', True),
            residual_connections=self.config['model'].get('residual_connections', True)
        )
        
        # Initialize model
        self.model = ImprovedLSTM(model_config)

    def prepare_datasets(self, historical_data: Dict[str, pd.DataFrame]) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation datasets
        
        Args:
            historical_data (Dict[str, pd.DataFrame]): Historical price data
            
        Returns:
            Tuple[DataLoader, DataLoader]: Train and validation data loaders
        """
        try:
            # Get features and create loaders
            X_train, X_val, y_train, y_val = self.feature_engineering.transform(
                historical_data, 
                train_size=self.config['model'].get('train_test_split', 0.8)
            )
            
            # Create datasets
            train_dataset = TimeSeriesDataset(X_train, y_train, self.sequence_length)
            val_dataset = TimeSeriesDataset(X_val, y_val, self.sequence_length)
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
            
            # Store input size for model initialization
            self.input_size = X_train.shape[2]
            
            return train_loader, val_loader
            
        except Exception as e:
            print(f"\n!!! Error during dataset preparation !!!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            traceback.print_exc()
            raise

    def train(self, train_loader: DataLoader, val_loader: DataLoader, force_rebuild: bool = False) -> None:
        """
        Train the model using prepared data loaders
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
        """
        print("\n====== Starting Model Training Process ======")
        print("\nModel Configuration:")
        print("----------------------------------------")
        print(f"Model Type: {self.config['model']['type']}")
        print(f"Model Version: {self.config['model']['version']}")
        print("\nArchitecture Parameters:")
        print(f"Input Size: {self.input_size}")
        print(f"Hidden Size: {self.hidden_size}")
        print(f"Number of Layers: {self.num_layers}")
        print(f"Sequence Length: {self.sequence_length}")
        print(f"Bidirectional: {self.config['model'].get('bidirectional', True)}")
        print(f"Attention Heads: {self.config['model'].get('attention_heads', 4)}")
        print(f"Layer Normalization: {self.config['model'].get('use_layer_norm', True)}")
        print(f"Residual Connections: {self.config['model'].get('residual_connections', True)}")
        
        print("\nTraining Parameters:")
        print(f"Batch Size: {self.batch_size}")
        print(f"Number of Epochs: {self.num_epochs}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Weight Decay: {self.config['model'].get('weight_decay', 0.01)}")
        print(f"Dropout Rate: {self.config['model'].get('dropout', 0.2)}")
        
        print("\nOptimizer Configuration:")
        print(f"Optimizer: AdamW")
        print(f"Learning Rate Scheduler: OneCycleLR")
        print(f"LR Scheduler Max LR: {self.config['model'].get('lr_scheduler', {}).get('max_lr', 0.01)}")
        print(f"LR Scheduler Pct Start: {self.config['model'].get('lr_scheduler', {}).get('pct_start', 0.3)}")
        
        print("\nEarly Stopping Configuration:")
        print(f"Patience: {self.config['model'].get('early_stopping_patience', 10)}")
        print(f"Min Delta: {self.config['model'].get('min_delta', 0.001)}")
        
        print("\nDataset Information:")
        print(f"Training Batches: {len(train_loader)}")
        print(f"Validation Batches: {len(val_loader)}")
        print(f"Samples per Batch: {self.batch_size}")
        total_train = len(train_loader) * self.batch_size
        total_val = len(val_loader) * self.batch_size
        print(f"Total Training Samples: {total_train}")
        print(f"Total Validation Samples: {total_val}")
        print("----------------------------------------")
        try:
            # Initialize model if not already initialized
            if self.model is None or force_rebuild:
                self._initialize_model()
            
            # Get optimizer and scheduler from model
            optimizer, scheduler = self.model.configure_optimizers(
                learning_rate=self.learning_rate,
                weight_decay=self.config['model'].get('weight_decay', 0.0001)
            )
            
            # Loss function
            criterion = nn.MSELoss()
            
            # Initialize best metrics
            best_val_metrics = {
                'loss': float('inf'),
                'direction_accuracy': 0,
                'sharpe_ratio': -float('inf'),
                'information_coefficient': -float('inf')
            }
            patience_counter = 0
            
            # Training loop
            for epoch in range(self.num_epochs):
                # Training phase
                self.model.train()
                train_metrics = self._train_epoch(train_loader, optimizer, criterion)
                
                # Validation phase
                self.model.eval()
                val_metrics = self._validate_epoch(val_loader, criterion)
                
                # Learning rate scheduling
                scheduler.step()
                
                # Early stopping check based on multiple metrics
                improvement = (
                    val_metrics['loss'] < best_val_metrics['loss'] - self.config['model'].get('min_delta', 0.001) or
                    val_metrics['direction_accuracy'] > best_val_metrics['direction_accuracy'] + 0.01 or
                    val_metrics['sharpe_ratio'] > best_val_metrics['sharpe_ratio'] + 0.1
                )
                
                if improvement:
                    best_val_metrics = val_metrics.copy()
                    patience_counter = 0
                    self._save_model()
                    print(f"New best model saved with metrics: {val_metrics}")
                else:
                    patience_counter += 1
                
                # Print epoch metrics
                if (epoch + 1) % 5 == 0:
                    print(f"\nEpoch [{epoch+1}/{self.num_epochs}]")
                    print(f"Train metrics: {train_metrics}")
                    print(f"Val metrics: {val_metrics}")
                
                # Early stopping check
                if patience_counter >= self.config['model'].get('early_stopping_patience', 10):
                    print(f'\nEarly stopping triggered at epoch {epoch+1}')
                    break
            
            print("\n=== Training Complete ===")
            print(f"Best validation metrics: {best_val_metrics}")
            
        except Exception as e:
            print(f"\n!!! Error during model training !!!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            traceback.print_exc()
            raise

    def _train_epoch(self, train_loader, optimizer, criterion):
        """Run one epoch of training"""
        total_loss = 0
        all_y_true = []
        all_y_pred = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            
            # Store predictions and actual values
            all_y_true.append(batch_y)
            all_y_pred.append(outputs.squeeze())
            
            # Calculate MSE loss
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Combine all batches
        y_true = torch.cat(all_y_true)
        y_pred = torch.cat(all_y_pred)
        
        # Calculate average loss and other metrics
        avg_loss = total_loss / len(train_loader)
        metrics = calculate_financial_metrics(y_true, y_pred)
        metrics['loss'] = avg_loss
        
        return metrics

    def _validate_epoch(self, val_loader, criterion):
        """Run one epoch of validation"""
        total_loss = 0
        all_y_true = []
        all_y_pred = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                
                # Store predictions and actual values
                all_y_true.append(batch_y)
                all_y_pred.append(outputs.squeeze())
                
                # Calculate MSE loss
                loss = criterion(outputs, batch_y.unsqueeze(1))
                total_loss += loss.item()
        
        # Combine all batches
        y_true = torch.cat(all_y_true)
        y_pred = torch.cat(all_y_pred)
        
        # Calculate average loss and other metrics
        avg_loss = total_loss / len(val_loader)
        metrics = calculate_financial_metrics(y_true, y_pred)
        metrics['loss'] = avg_loss
        
        return metrics

    def _save_model(self) -> None:
        """Save model, metadata, and scaler"""
        try:
            # Save model
            torch.save(self.model.state_dict(), self.model_path)
            
            # Save metadata with config fingerprint and full configuration
            metadata = {
                'version': self.config['model']['version'],
                'training_date': datetime.now().isoformat(),
                'model_type': 'ImprovedLSTM',
                'config_fingerprint': self.config_fingerprint,
                'parameters': {
                    'architecture': {
                        'input_size': self.config['model']['input_size'],
                        'hidden_size': self.hidden_size,
                        'num_layers': self.num_layers,
                        'bidirectional': self.config['model'].get('bidirectional', True),
                        'attention_heads': self.config['model'].get('attention_heads', 4),
                        'use_layer_norm': self.config['model'].get('use_layer_norm', True),
                        'residual_connections': self.config['model'].get('residual_connections', True)
                    },
                    'sequence_length': self.sequence_length,
                    'features': self.config.get('features', {})  # Save the entire features config
                },
                'feature_engineering': self.feature_engineering.get_config()  # Add this line to save feature engineering state
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # Save scaler
            if self.scaler is not None:
                joblib.dump(self.scaler, self.scaler_path)
            
            self.logger.info("Successfully saved model and metadata")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def predict(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Generate predictions for input data"""
        self.logger.info(f"Starting prediction process for {len(data)} symbols")
        
        try:
            # Load model if not loaded
            if self.model is None:
                self._load_model()
                if self.model is None:
                    raise ValueError("No compatible trained model available")
            
            # Safely get feature configuration from metadata
            if hasattr(self, 'metadata'):
                feature_config = (
                    self.metadata.get('feature_engineering') or 
                    self.metadata.get('parameters', {}).get('features', {})
                )
                if feature_config:
                    self.feature_engineering.update_config(feature_config)
                else:
                    self.logger.warning("No feature configuration found in metadata, using current config")
            
            # Feature engineering with error handling
            try:
                features = self.feature_engineering.transform_predict(data)
            except Exception as e:
                self.logger.error(f"Error during feature engineering: {str(e)}")
                raise
            
            # Make predictions
            self.model.eval()
            predictions = {}
            
            with torch.no_grad():
                for symbol in features.keys():  # Only iterate over symbols that have features
                    try:
                        # Get features for this symbol
                        symbol_features = torch.FloatTensor(features[symbol]).unsqueeze(0)
                        
                        # Generate prediction
                        pred = self.model(symbol_features)
                        predictions[symbol] = pd.Series(
                            pred.numpy().flatten(),
                            index=data[symbol].index[-len(pred):]
                        )
                    except Exception as e:
                        self.logger.error(f"Error predicting for symbol {symbol}: {str(e)}")
                        continue
            
            if not predictions:
                raise ValueError("No valid predictions could be generated")
                
            return predictions
                
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            Dict[str, float]: Training metrics
        """
        # Initialize model for training
        self.model.train()
        
        # Initialize optimizer and criterion
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.config['model'].get('weight_decay', 0.0001)
        )
        criterion = nn.MSELoss()
        
        # Train one epoch
        metrics = self._train_epoch(train_loader, optimizer, criterion)
        return metrics

    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate the model
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            Dict[str, float]: Validation metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize criterion
        criterion = nn.MSELoss()
        
        # Run validation
        metrics = self._validate_epoch(val_loader, criterion)
        return metrics

    def _load_model(self) -> None:
        """Load saved model and validate against current configuration"""
        try:
            if os.path.exists(self.model_path):
                print("\nFound existing model checkpoint")
                
                # Load metadata first
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                try:
                    # Create model config from metadata (not current config)
                    model_config = LSTMConfig(**self.metadata['parameters']['architecture'])
                    
                    # Initialize and load model
                    self.model = ImprovedLSTM(model_config)
                    self.model.load_state_dict(torch.load(self.model_path))
                    self.model.eval()
                    
                    # Load scaler
                    if os.path.exists(self.scaler_path):
                        self.scaler = joblib.load(self.scaler_path)
                    
                    print("Successfully loaded existing model and metadata")
                    
                except Exception as e:
                    print(f"Warning: Could not load existing model due to: {str(e)}")
                    print("Will train a new model instead")
                    self._cleanup_model_files()
                    self.model = None
            else:
                print("\nNo existing model found - will train a new one")
                self.model = None
                    
        except Exception as e:
            print(f"\nError during model loading: {str(e)}")
            print("Will train a new model instead")
            self.model = None

    def _cleanup_model_files(self):
        """Remove existing model files"""
        for path in [self.model_path, self.metadata_path, self.scaler_path]:
            if os.path.exists(path):
                os.remove(path)

    def optimize_hyperparameters(self, historical_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Find optimal hyperparameters for the model
        
        Args:
            historical_data (Dict[str, pd.DataFrame]): Historical price data
            
        Returns:
            Dict: Best hyperparameters found
        """
        print("\n=== Starting Hyperparameter Optimization Process ===")
        
        try:
            # Create optimizer
            optimizer = HyperparameterOptimizer(self.config)
            
            # Prepare datasets once for all trials
            train_loader, val_loader = self.prepare_datasets(historical_data)
            
            # Run optimization
            best_params, best_metrics = optimizer.optimize(
                self,
                train_loader,
                val_loader,
                seed=self.config.get('optimization', {}).get('seed', None)
            )
            
            # Update model configuration with best parameters
            self.config['model'].update(best_params)
            
            # Log optimization results
            self.logger.info(f"Best parameters found: {best_params}")
            self.logger.info(f"Best metrics achieved: {best_metrics}")
            
            # Reset model to use new parameters
            self.model = None  # Force reinitialization with new parameters
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"Error during hyperparameter optimization: {str(e)}")
            raise
