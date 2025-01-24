import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import os
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pyarrow.parquet as pq

class BacktestDataManager:
    """
    Manages data retrieval and caching for backtesting operations.
    Implements an efficient caching system and provides validated market data.
    """
    
    def __init__(self, db_path: str, cache_dir: str):
        """
        Initialize the BacktestDataManager with database and cache paths.
        
        Args:
            db_path (str): Path to SQLite database
            cache_dir (str): Path to cache directory
        """
        self.db_path = db_path
        self.cache_dir = cache_dir
        self.memory_cache: Dict[str, pd.DataFrame] = {}
        self.invalid_symbols: Dict[str, datetime] = {}
        
        # Ensure directories exist
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(cache_dir, 'metadata')).mkdir(exist_ok=True)
        Path(os.path.join(cache_dir, 'daily_data')).mkdir(exist_ok=True)
        Path(os.path.join(cache_dir, 'temp', 'downloads')).mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize database
        self._initialize_database()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self) -> dict:
        """Load configuration settings from YAML file."""
        config_path = os.path.join('config', 'config.yaml')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                print(f"Loaded configuration from {config_path}")
                return config
        else:
            raise FileNotFoundError(f"Configuration file not found at {config_path}. Please ensure config.yaml exists.")

    def _initialize_database(self) -> None:
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS symbols_meta (
                    symbol TEXT PRIMARY KEY,
                    first_available_date DATE,
                    last_available_date DATE,
                    last_updated TIMESTAMP,
                    is_valid BOOLEAN
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_data (
                    symbol TEXT,
                    date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, date)
                )
            """)

    def get_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Primary method to retrieve data for backtesting.
        
        Args:
            symbols (List[str]): List of stock symbols
            start_date (datetime): Start date for data
            end_date (datetime): End date for data
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data
        """
        print(f"\n=== Starting get_data ===")
        print(f"Input symbols: {symbols}")
        print(f"Date range: {start_date} to {end_date}")
        
        # Validate symbols first
        print("\nValidating symbols...")
        valid_symbols = self._validate_symbols(symbols, start_date, end_date)
        print(f"Valid symbols after validation: {valid_symbols}")
        
        if not valid_symbols:
            print("WARNING: No valid symbols found after validation!")
            return {}
        
        # Initialize result dictionary
        result = {}
        
        print("\nStarting parallel processing...")
        # Process symbols in parallel
        with ThreadPoolExecutor(max_workers=self.config['download']['batch_size']) as executor:
            print(f"Created ThreadPoolExecutor with max_workers: {self.config['download']['batch_size']}")
            
            future_to_symbol = {
                executor.submit(self._get_symbol_data, symbol, start_date, end_date): symbol
                for symbol in valid_symbols
            }
            print(f"Submitted {len(future_to_symbol)} tasks to executor")
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                print(f"\nProcessing result for symbol: {symbol}")
                try:
                    data = future.result()
                    if data is not None:
                        print(f"Successfully retrieved data for {symbol}, shape: {data.shape}")
                        result[symbol] = data
                    else:
                        print(f"No data returned for {symbol}")
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")
                    self.logger.error(f"Error processing {symbol}: {str(e)}")
        
        print(f"\n=== Completed get_data ===")
        print(f"Successfully retrieved data for {len(result)} symbols: {list(result.keys())}")
        return result

    def _get_symbol_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get data for a single symbol with caching."""
        # Check memory cache first
        if symbol in self.memory_cache:
            df = self.memory_cache[symbol]
            if self._is_data_valid_for_range(df, start_date, end_date):
                return df
        
        # Check disk cache
        cache_path = self._get_cache_path(symbol)
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            if self._is_data_valid_for_range(df, start_date, end_date):
                # Update memory cache
                self.memory_cache[symbol] = df
                return df
        
        # Download if not in cache
        try:
            df = self._download_data(symbol, start_date, end_date)
            if df is not None:
                # Save to cache
                self._save_to_cache(symbol, df)
                return df
        except Exception as e:
            self.logger.error(f"Failed to download {symbol}: {str(e)}")
            return None

    def _validate_symbols(self, symbols: List[str], start_date: datetime, end_date: datetime) -> List[str]:
        """
        Validate symbols and filter out invalid ones.
        
        Args:
            symbols (List[str]): List of symbols to validate
            start_date (datetime): Start date for validation
            end_date (datetime): End date for validation
            
        Returns:
            List[str]: List of valid symbols
        """
        print(f"\n=== Starting symbol validation ===")
        print(f"Validating {len(symbols)} symbols: {symbols}")
        valid_symbols = []
        
        for symbol in symbols:
            # Check invalid symbols cache
            if symbol in self.invalid_symbols:
                cache_age = datetime.now() - self.invalid_symbols[symbol]
                print(f"Symbol found in invalid cache, age: {cache_age}")
                if cache_age < timedelta(days=self.config['cache']['cache_expiry_days']):
                    print(f"Skipping {symbol} - marked as invalid in cache")
                    continue
                else:
                    print(f"Invalid cache expired for {symbol}, will revalidate")
            
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT is_valid, first_available_date, last_available_date
                        FROM symbols_meta
                        WHERE symbol = ?
                    """, (symbol,))
                    
                    result = cursor.fetchone()
                    print(f"Database query result for {symbol}: {result}")
                    
                    if result:  # Symbol exists in database
                        is_valid = result[0]
                        if not is_valid:
                            print(f"Symbol {symbol} marked as invalid in database")
                            continue
                            
                        first_date = datetime.strptime(result[1], '%Y-%m-%d').date()
                        last_date = datetime.strptime(result[2], '%Y-%m-%d').date()
                        print(f"Date range in DB: {first_date} to {last_date}")
                        
                        # If we have data that partially covers our range, consider it valid
                        # We'll download new data as needed
                        if not (first_date > end_date.date() or last_date < start_date.date()):
                            print(f"Adding {symbol} to valid symbols - date range acceptable")
                            valid_symbols.append(symbol)
                        else:
                            print(f"Date range mismatch for {symbol}")
                    else:
                        # New symbol - always consider valid for first attempt
                        print(f"New symbol {symbol} - adding to valid symbols for first attempt")
                        valid_symbols.append(symbol)
                        
            except Exception as e:
                print(f"Error checking database for {symbol}: {str(e)}")
                # If there's an error checking the database, we'll try to validate the symbol
                print(f"Adding {symbol} to valid symbols despite error")
                valid_symbols.append(symbol)
        
        print(f"\n=== Completed symbol validation ===")
        print(f"Found {len(valid_symbols)} valid symbols: {valid_symbols}")
        return valid_symbols

    def _download_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Download data from yfinance with retry logic.
        
        Args:
            symbol (str): Stock symbol
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            Optional[pd.DataFrame]: Downloaded data or None if failed
        """
        print(f"\n=== Downloading data for {symbol} ===")
        print(f"Date range: {start_date} to {end_date}")
        
        for attempt in range(self.config['download']['max_retries']):
            try:
                print(f"Download attempt {attempt + 1} for {symbol}")
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval='1d'
                )
                
                print(f"Downloaded {len(df)} rows of data for {symbol}")
                
                if df.empty:
                    print(f"No data returned for {symbol}")
                    return None
                    
                if len(df) >= self.config['validation']['min_data_points']:
                    print(f"Sufficient data points ({len(df)}) for {symbol}")
                    df = self._validate_and_clean_data(df)
                    if df is not None:
                        print(f"Data validation successful for {symbol}")
                        self._update_symbol_metadata(symbol, df)
                        return df
                    else:
                        print(f"Data validation failed for {symbol}")
                else:
                    print(f"Insufficient data points ({len(df)}) for {symbol}")
                
                return None
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                if attempt < self.config['download']['max_retries'] - 1:
                    print(f"Waiting {self.config['download']['retry_delay']} seconds before retry...")
                    time.sleep(self.config['download']['retry_delay'])
        
        print(f"All attempts failed for {symbol}, marking as invalid")
        self.invalid_symbols[symbol] = datetime.now()
        return None

    def _validate_and_clean_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Validate and clean downloaded data.
        
        Args:
            df (pd.DataFrame): Raw data
            
        Returns:
            Optional[pd.DataFrame]: Cleaned data or None if invalid
        """
        if df.empty:
            return None
            
        # Check for missing values
        missing_pct = df.isnull().mean().max()
        if missing_pct > self.config['validation']['max_missing_pct']:
            return None
            
        # Fill missing values
        df = df.ffill().bfill()
        
        # Check for price validity
        if (df['Close'] < self.config['validation']['price_threshold']).any():
            return None
            
        return df

    def _update_symbol_metadata(self, symbol: str, df: pd.DataFrame) -> None:
        """Update symbol metadata in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO symbols_meta
                (symbol, first_available_date, last_available_date, last_updated, is_valid)
                VALUES (?, ?, ?, ?, ?)
            """, (
                symbol,
                df.index[0].date().isoformat(),
                df.index[-1].date().isoformat(),
                datetime.now().isoformat(),
                True
            ))

    def _save_to_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(symbol)
        df.to_parquet(cache_path)
        
        # Update memory cache
        if len(self.memory_cache) >= self.config['cache']['max_memory_cache_size']:
            # Remove oldest item
            oldest_symbol = next(iter(self.memory_cache))
            del self.memory_cache[oldest_symbol]
        
        self.memory_cache[symbol] = df

    def _get_cache_path(self, symbol: str) -> str:
        """Get cache file path for symbol."""
        return os.path.join(
            self.cache_dir,
            'daily_data',
            f"{symbol}.parquet"
        )

    def _is_data_valid_for_range(self, df: pd.DataFrame, start_date: datetime, end_date: datetime) -> bool:
        """Check if cached data covers the requested date range."""
        if df.empty:
            return False
            
        return (df.index[0].date() <= start_date.date() and 
                df.index[-1].date() >= end_date.date())

    def update_cache(self, symbols: List[str]) -> None:
        """
        Update cache for specified symbols.
        
        Args:
            symbols (List[str]): List of symbols to update
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config['cache']['cache_expiry_days'])
        
        with ThreadPoolExecutor(max_workers=self.config['download']['batch_size']) as executor:
            future_to_symbol = {
                executor.submit(self._download_data, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Failed to update cache for {symbol}: {str(e)}")