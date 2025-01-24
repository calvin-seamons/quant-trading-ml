import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import yaml
import os
from models.LSTMManager import LSTMManager

@dataclass
class Position:
    """Dataclass to store position information"""
    symbol: str
    score: float
    size: float
    direction: str  # 'long' or 'short'
    risk_metrics: Dict

class MeanReversion:
    """Stub class for mean reversion signals"""
    def __init__(self, config: Dict):
        self.config = config
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Stub for mean reversion signals"""
        return pd.Series()

class FactorPipeline:
    """
    Pipeline for analyzing and ranking trading opportunities based on 
    multiple factors including ML predictions and mean reversion signals.
    """
    
    def __init__(self, historical_data: Dict[str, pd.DataFrame]):
        """
        Initialize the factor pipeline with historical data and configuration.
        
        Args:
            historical_data (Dict[str, pd.DataFrame]): Dictionary of price data by symbol
        """
        self.logger = logging.getLogger(__name__)
        self.historical_data = historical_data
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize storage
        self.filtered_universe: List[str] = []
        self.long_positions: List[Position] = []
        self.short_positions: List[Position] = []
        
        # Initialize factor models
        self.ml_model = LSTMManager()
        self.mean_reversion = MeanReversion(self.config)
        
        # Store latest analysis results
        self.latest_scores: Dict[str, float] = {}
        self.latest_risk_metrics: Dict[str, Dict] = {}

    def _load_config(self) -> Dict:
        """
        Load configuration from YAML file.
        
        Returns:
            Dict: Configuration dictionary
        """
        config_path = os.path.join('config', 'config.yaml')
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.logger.info(f"Successfully loaded configuration from {config_path}")
                return config
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found at {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing configuration file: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading configuration: {str(e)}")
            raise
        
    def basic_screen(self) -> List[str]:
        """
        Apply basic screening criteria to filter universe.
        
        Returns:
            List[str]: List of symbols that pass basic screening
        """
        filtered_symbols = []
        min_price = self.config['validation']['price_threshold']
        min_volume = 100000  # 100K daily average volume
        
        for symbol, df in self.historical_data.items():
            try:
                # Calculate average price and volume
                avg_price = df['Close'].mean()
                avg_volume = df['Volume'].mean()
                
                # Apply filters
                if avg_price >= min_price and avg_volume >= min_volume:
                    filtered_symbols.append(symbol)
                    
            except Exception as e:
                self.logger.error(f"Error screening {symbol}: {str(e)}")
                
        self.filtered_universe = filtered_symbols
        return filtered_symbols
        
    def analyze_stocks(self) -> Dict[str, float]:
        """
        Analyze stocks using multiple factors and combine signals.
        
        Returns:
            Dict[str, float]: Dictionary of composite scores by symbol
        """
        composite_scores = {}
        
        for symbol in self.filtered_universe:
            try:
                df = self.historical_data[symbol]
                
                # Get ML predictions
                ml_score = self.ml_model.predict(df)
                
                # Get mean reversion signals
                mr_score = self.mean_reversion.calculate_signals(df)
                
                # Combine scores using weights from config
                ml_weight = self.config['factor_pipeline']['model_weight']
                mr_weight = self.config['factor_pipeline']['mean_reversion_weight']
                
                # For now, just use the last value of each signal
                composite_score = (
                    ml_weight * ml_score.iloc[-1] +
                    mr_weight * mr_score.iloc[-1]
                )
                
                composite_scores[symbol] = composite_score
                
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {str(e)}")
                
        self.latest_scores = composite_scores
        return composite_scores
        
    def rank_opportunities(self) -> Tuple[List[Position], List[Position]]:
        """
        Rank opportunities and calculate position sizes.
        
        Returns:
            Tuple[List[Position], List[Position]]: Ranked long and short positions
        """
        positions = []
        
        # Configuration parameters
        max_position_size = self.config['portfolio_strategy']['max_position_size']
        min_score = self.config['factor_pipeline']['min_score_threshold']
        
        for symbol, score in self.latest_scores.items():
            try:
                # Determine direction based on score
                direction = 'long' if score > min_score else 'short'
                
                # Calculate base position size
                size = min(abs(score), max_position_size)
                
                # Calculate risk metrics (stub for now)
                risk_metrics = self._calculate_risk_metrics(symbol)
                
                # Create position object
                position = Position(
                    symbol=symbol,
                    score=score,
                    size=size,
                    direction=direction,
                    risk_metrics=risk_metrics
                )
                
                positions.append(position)
                
            except Exception as e:
                self.logger.error(f"Error ranking {symbol}: {str(e)}")
        
        # Sort positions by absolute score
        positions.sort(key=lambda x: abs(x.score), reverse=True)
        
        # Split into long and short lists
        self.long_positions = [p for p in positions if p.direction == 'long']
        self.short_positions = [p for p in positions if p.direction == 'short']
        
        return self.long_positions, self.short_positions
    
    def _calculate_risk_metrics(self, symbol: str) -> Dict:
        """
        Calculate risk metrics for position sizing (stub for now).
        
        Args:
            symbol (str): Symbol to calculate metrics for
            
        Returns:
            Dict: Dictionary of risk metrics
        """
        # Stub implementation - will be expanded later
        return {
            'volatility': 0.0,
            'beta': 1.0,
            'sector_exposure': 0.0
        }
    
    def get_filtered_universe(self) -> List[str]:
        """
        Get the current filtered universe of symbols.
        
        Returns:
            List[str]: List of symbols in filtered universe
        """
        return self.filtered_universe
    
    def get_position_data(self) -> Dict[str, List[Position]]:
        """
        Get current position recommendations.
        
        Returns:
            Dict[str, List[Position]]: Dictionary containing long and short positions
        """
        return {
            'long': self.long_positions,
            'short': self.short_positions
        }