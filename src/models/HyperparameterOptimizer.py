import optuna
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime
import torch
from sklearn.model_selection import TimeSeriesSplit
import yaml
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import joblib

@dataclass
class OptimizationMetrics:
    """Container for optimization metrics"""
    sharpe_ratio: float
    max_drawdown: float
    hit_ratio: float
    avg_return: float
    volatility: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'hit_ratio': self.hit_ratio,
            'avg_return': self.avg_return,
            'volatility': self.volatility
        }

class HyperparameterOptimizer:
    """
    Handles hyperparameter optimization for the LSTM trading model.
    Implements multi-phase optimization with Bayesian search and parallel processing.
    """
    
    def __init__(self, lstm_manager, config: Dict[str, Any]) -> None:
        """
        Initialize the optimizer with an LSTM manager instance and configuration.
        
        Args:
            lstm_manager: Instance of LSTMManager class
            config: Configuration dictionary containing optimization parameters
        """
        self.lstm_manager = lstm_manager
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize study storage
        self.storage_path = Path("optimization/studies")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results_path = Path("optimization/results")
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Validation data splits
        self.splits = None
        
        # Best parameters found
        self.best_params = None
        
        self.logger.info("HyperparameterOptimizer initialized")

    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the optimizer"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler('optimization.log')
        console_handler = logging.StreamHandler()
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def setup_search_space(self) -> Dict[str, Any]:
        """
        Define the search space for different parameters based on optimization phase.
        
        Returns:
            Dictionary containing parameter ranges for each optimization phase
        """
        search_space = {
            'coarse_search': {
                'sequence_length': (10, 100),
                'hidden_size': (32, 256),
                'num_layers': (1, 4),
                'learning_rate': (1e-5, 1e-2, 'log')
            },
            'feature_optimization': {
                'price_window': (5, 50),
                'volume_window': (5, 50),
                'momentum_window': (10, 100),
                'volatility_window': (10, 100)
            },
            'fine_tuning': {
                'dropout': (0.1, 0.5),
                'attention_heads': (1, 8),
                'batch_size': (16, 128),
                'weight_decay': (1e-6, 1e-3, 'log')
            }
        }
        
        self.logger.info(f"Search space configured: {json.dumps(search_space, indent=2)}")
        return search_space

    def setup_validation_strategy(self, data: pd.DataFrame) -> None:
        """
        Configure the validation strategy for hyperparameter optimization.
        
        Args:
            data: Input DataFrame with financial data
        """
        n_splits = self.config.get('validation', {}).get('n_splits', 5)
        gap = self.config.get('validation', {}).get('gap', 5)
        
        # Create TimeSeriesSplit with gap
        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            gap=gap,
            test_size=int(len(data) * 0.2)
        )
        
        # Store splits for later use
        self.splits = list(tscv.split(data))
        
        self.logger.info(f"Validation strategy configured with {n_splits} splits and {gap} day gap")

    def run_coarse_search(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform initial broad parameter search using Bayesian optimization.
        
        Args:
            data: Input DataFrame with financial data
            
        Returns:
            Dictionary containing best parameters from coarse search
        """
        self.logger.info("Starting coarse parameter search")
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                'sequence_length': trial.suggest_int('sequence_length', 10, 100),
                'hidden_size': trial.suggest_int('hidden_size', 32, 256),
                'num_layers': trial.suggest_int('num_layers', 1, 4),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
            }
            
            return self._evaluate_parameters(params, data, 'coarse')
        
        study = optuna.create_study(
            study_name='coarse_search',
            direction='maximize',
            storage=f'sqlite:///{self.storage_path}/coarse_search.db',
            load_if_exists=True
        )
        
        n_trials = self.config.get('coarse_search', {}).get('n_trials', 100)
        
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=self.config.get('resources', {}).get('n_parallel', 1)
        )
        
        best_params = study.best_params
        self.logger.info(f"Coarse search completed. Best parameters: {best_params}")
        
        # Save results
        self._save_optimization_results('coarse_search', study)
        
        return best_params

    def optimize_features(self, data: pd.DataFrame, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize feature engineering parameters using the best base parameters.
        
        Args:
            data: Input DataFrame with financial data
            base_params: Best parameters from coarse search
            
        Returns:
            Dictionary containing optimized feature parameters
        """
        self.logger.info("Starting feature optimization")
        
        def objective(trial: optuna.Trial) -> float:
            feature_params = {
                'price_window': trial.suggest_int('price_window', 5, 50),
                'volume_window': trial.suggest_int('volume_window', 5, 50),
                'momentum_window': trial.suggest_int('momentum_window', 10, 100),
                'volatility_window': trial.suggest_int('volatility_window', 10, 100)
            }
            
            # Combine with base parameters
            params = {**base_params, **feature_params}
            
            return self._evaluate_parameters(params, data, 'features')
        
        study = optuna.create_study(
            study_name='feature_optimization',
            direction='maximize',
            storage=f'sqlite:///{self.storage_path}/feature_optimization.db',
            load_if_exists=True
        )
        
        n_trials = self.config.get('feature_optimization', {}).get('n_trials', 50)
        
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=self.config.get('resources', {}).get('n_parallel', 1)
        )
        
        best_params = {**base_params, **study.best_params}
        self.logger.info(f"Feature optimization completed. Best parameters: {best_params}")
        
        # Save results
        self._save_optimization_results('feature_optimization', study)
        
        return best_params

    def fine_tune_parameters(self, data: pd.DataFrame, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fine-tune model parameters using the best parameters from previous phases.
        
        Args:
            data: Input DataFrame with financial data
            base_params: Best parameters from previous optimization phases
            
        Returns:
            Dictionary containing final optimized parameters
        """
        self.logger.info("Starting fine-tuning phase")
        
        def objective(trial: optuna.Trial) -> float:
            fine_tune_params = {
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'attention_heads': trial.suggest_int('attention_heads', 1, 8),
                'batch_size': trial.suggest_int('batch_size', 16, 128),
                'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
            }
            
            # Combine with base parameters
            params = {**base_params, **fine_tune_params}
            
            return self._evaluate_parameters(params, data, 'fine_tune')
        
        study = optuna.create_study(
            study_name='fine_tuning',
            direction='maximize',
            storage=f'sqlite:///{self.storage_path}/fine_tuning.db',
            load_if_exists=True
        )
        
        n_trials = self.config.get('fine_tuning', {}).get('n_trials', 200)
        
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=self.config.get('resources', {}).get('n_parallel', 1)
        )
        
        self.best_params = {**base_params, **study.best_params}
        self.logger.info(f"Fine-tuning completed. Final parameters: {self.best_params}")
        
        # Save results
        self._save_optimization_results('fine_tuning', study)
        
        return self.best_params

    def run_robustness_tests(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test robustness of optimized parameters across different market conditions.
        
        Args:
            data: Input DataFrame with financial data
            
        Returns:
            Dictionary containing robustness metrics
        """
        self.logger.info("Starting robustness testing")
        
        if self.best_params is None:
            raise ValueError("No optimized parameters available for robustness testing")
        
        # Define market regimes
        regimes = self._identify_market_regimes(data)
        
        results = {}
        
        # Test across different regimes
        for regime, regime_data in regimes.items():
            self.logger.info(f"Testing robustness in {regime} regime")
            
            metrics = self._evaluate_parameters(
                self.best_params,
                regime_data,
                'robustness',
                regime=regime
            )
            
            results[regime] = metrics
        
        # Save robustness results
        self._save_robustness_results(results)
        
        return results

    def _evaluate_parameters(
        self,
        params: Dict[str, Any],
        data: pd.DataFrame,
        phase: str,
        regime: Optional[str] = None
    ) -> float:
        """
        Evaluate a set of parameters using walk-forward validation.
        
        Args:
            params: Parameters to evaluate
            data: Input data
            phase: Optimization phase
            regime: Optional market regime for robustness testing
            
        Returns:
            float: Evaluation metric (e.g., Sharpe ratio)
        """
        metrics_list = []
        
        for train_idx, val_idx in self.splits:
            # Split data
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]
            
            try:
                # Configure model with parameters
                self.lstm_manager.reconfigure(params)
                
                # Train model
                train_metrics = self.lstm_manager.train(train_data)
                
                # Evaluate on validation set
                val_metrics = self._calculate_metrics(val_data)
                metrics_list.append(val_metrics)
                
            except Exception as e:
                self.logger.warning(f"Parameter evaluation failed: {str(e)}")
                return float('-inf')
        
        # Average metrics across folds
        avg_metrics = self._average_metrics(metrics_list)
        
        # Log evaluation
        self.logger.info(
            f"Parameter evaluation - Phase: {phase}, "
            f"Regime: {regime if regime else 'all'}, "
            f"Metrics: {avg_metrics}"
        )
        
        # Return primary metric (e.g., Sharpe ratio)
        return avg_metrics.sharpe_ratio

    def _calculate_metrics(self, data: pd.DataFrame) -> OptimizationMetrics:
        """Calculate performance metrics for a dataset"""
        predictions = self.lstm_manager.predict(data)
        returns = self._calculate_returns(predictions, data)
        
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        max_drawdown = self._calculate_max_drawdown(returns)
        hit_ratio = (returns > 0).mean()
        avg_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        
        return OptimizationMetrics(
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            hit_ratio=hit_ratio,
            avg_return=avg_return,
            volatility=volatility
        )

    def _calculate_returns(self, predictions: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Calculate returns based on predictions"""
        # Implementation depends on how predictions are used to generate returns
        pass

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = cumulative / running_max - 1
        return drawdowns.min()

    def _average_metrics(self, metrics_list: List[OptimizationMetrics]) -> OptimizationMetrics:
        """Average metrics across multiple evaluations"""
        n = len(metrics_list)
        return OptimizationMetrics(
            sharpe_ratio=sum(m.sharpe_ratio for m in metrics_list) / n,
            max_drawdown=sum(m.max_drawdown for m in metrics_list) / n,
            hit_ratio=sum(m.hit_ratio for m in metrics_list) / n,
            avg_return=sum(m.avg_return for m in metrics_list) / n,
            volatility=sum(m.volatility for m in metrics_list) / n
        )
        
    def _identify_market_regimes(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Identify different market regimes in the data.
        
        Args:
            data: Input DataFrame with financial data
            
        Returns:
            Dictionary mapping regime names to corresponding data segments
        """
        # Calculate returns
        returns = data['Close'].pct_change()
        
        # Calculate volatility using rolling window
        volatility = returns.rolling(window=21).std() * np.sqrt(252)
        
        # Calculate trend using moving averages
        sma_50 = data['Close'].rolling(window=50).mean()
        sma_200 = data['Close'].rolling(window=200).mean()
        
        # Define regimes
        regimes = {}
        
        # High volatility regime
        high_vol_mask = volatility > volatility.quantile(0.75)
        regimes['high_volatility'] = data[high_vol_mask]
        
        # Trending up regime
        trend_up_mask = (sma_50 > sma_200) & ~high_vol_mask
        regimes['trending_up'] = data[trend_up_mask]
        
        # Trending down regime
        trend_down_mask = (sma_50 < sma_200) & ~high_vol_mask
        regimes['trending_down'] = data[trend_down_mask]
        
        # Ranging regime (everything else)
        ranging_mask = ~(high_vol_mask | trend_up_mask | trend_down_mask)
        regimes['ranging'] = data[ranging_mask]
        
        return regimes
    
    def _save_optimization_results(self, phase: str, study: optuna.Study) -> None:
        """
        Save optimization results to disk.
        
        Args:
            phase: Optimization phase name
            study: Completed optuna study
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_path / f"{phase}_results_{timestamp}.json"
        
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study_name': study.study_name,
            'timestamp': timestamp
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        self.logger.info(f"Saved {phase} results to {results_file}")
        
    def _save_robustness_results(self, results: Dict[str, OptimizationMetrics]) -> None:
        """
        Save robustness testing results to disk.
        
        Args:
            results: Dictionary mapping regimes to their metrics
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_path / f"robustness_results_{timestamp}.json"
        
        # Convert metrics to dictionary format
        formatted_results = {
            regime: metrics.to_dict()
            for regime, metrics in results.items()
        }
        
        with open(results_file, 'w') as f:
            json.dump(formatted_results, f, indent=4)
            
        self.logger.info(f"Saved robustness results to {results_file}")
        
    def generate_optimization_report(self) -> str:
        """
        Generate a comprehensive report of the optimization process.
        
        Returns:
            Path to the generated report file
        """
        if self.best_params is None:
            raise ValueError("No optimization results available for report generation")
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.results_path / f"optimization_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Hyperparameter Optimization Report\n\n")
            
            # Summary
            f.write("## Summary\n")
            f.write(f"- Timestamp: {timestamp}\n")
            f.write(f"- Total phases completed: 3\n")
            f.write(f"- Best parameters found: {json.dumps(self.best_params, indent=2)}\n\n")
            
            # Phase details
            for phase in ['coarse_search', 'feature_optimization', 'fine_tuning']:
                f.write(f"## {phase.replace('_', ' ').title()}\n")
                phase_files = list(self.results_path.glob(f"{phase}_results_*.json"))
                if phase_files:
                    latest_file = max(phase_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_file) as pf:
                        phase_results = json.load(pf)
                    f.write(f"- Best value: {phase_results['best_value']}\n")
                    f.write(f"- Number of trials: {phase_results['n_trials']}\n")
                    f.write(f"- Parameters: {json.dumps(phase_results['best_params'], indent=2)}\n\n")
            
            # Robustness results
            robustness_files = list(self.results_path.glob("robustness_results_*.json"))
            if robustness_files:
                f.write("## Robustness Analysis\n")
                latest_file = max(robustness_files, key=lambda x: x.stat().st_mtime)
                with open(latest_file) as rf:
                    robustness_results = json.load(rf)
                for regime, metrics in robustness_results.items():
                    f.write(f"\n### {regime.replace('_', ' ').title()} Regime\n")
                    for metric, value in metrics.items():
                        f.write(f"- {metric}: {value:.4f}\n")
        
        self.logger.info(f"Generated optimization report: {report_file}")
        return str(report_file)