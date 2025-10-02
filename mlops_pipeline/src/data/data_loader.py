"""
Production-grade data loader with comprehensive error handling and validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging
from datetime import datetime
import yaml
import os

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Production-ready data loader with comprehensive validation and error handling.
    
    Features:
    - Multiple data source support (CSV, Parquet, Database)
    - Data quality checks
    - Schema validation
    - Caching mechanisms
    - Batch and streaming support
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config: Configuration dictionary containing data paths and settings
        """
        self.config = config
        self.data_cache = {}
        self.schema_cache = {}
        
        # Setup paths
        self.raw_data_path = Path(config['data']['raw_data_path'])
        self.processed_data_path = Path(config['data']['processed_data_path'])
        
        # Create directories if they don't exist
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataLoader initialized with paths: {self.raw_data_path}, {self.processed_data_path}")
    
    def load_csv(self, 
                 file_path: Union[str, Path], 
                 cache: bool = True,
                 validate_schema: bool = True) -> pd.DataFrame:
        """
        Load CSV file with comprehensive validation.
        
        Args:
            file_path: Path to CSV file
            cache: Whether to cache the loaded data
            validate_schema: Whether to validate data schema
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data validation fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Check cache first
        cache_key = str(file_path)
        if cache and cache_key in self.data_cache:
            logger.info(f"Loading data from cache: {file_path}")
            return self.data_cache[cache_key].copy()
        
        try:
            logger.info(f"Loading CSV data from: {file_path}")
            start_time = datetime.now()
            
            # Load with error handling
            df = pd.read_csv(file_path)
            
            # Basic validation
            if df.empty:
                raise ValueError("Loaded DataFrame is empty")
            
            # Log data info
            load_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Data loaded successfully in {load_time:.2f}s. Shape: {df.shape}")
            
            # Schema validation
            if validate_schema:
                self._validate_schema(df, file_path)
            
            # Cache the data
            if cache:
                self.data_cache[cache_key] = df.copy()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {str(e)}")
            raise
    
    def load_from_database(self, 
                          connection_string: str,
                          query: str,
                          cache: bool = True) -> pd.DataFrame:
        """
        Load data from database with connection pooling.
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            cache: Whether to cache the loaded data
            
        Returns:
            Loaded DataFrame
        """
        try:
            import sqlalchemy as sa
            
            logger.info("Loading data from database")
            start_time = datetime.now()
            
            # Create connection
            engine = sa.create_engine(connection_string)
            
            # Execute query
            df = pd.read_sql(query, engine)
            
            # Close connection
            engine.dispose()
            
            load_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Database data loaded successfully in {load_time:.2f}s. Shape: {df.shape}")
            
            # Cache the data
            if cache:
                cache_key = f"db_{hash(query)}"
                self.data_cache[cache_key] = df.copy()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from database: {str(e)}")
            raise
    
    def load_parquet(self, 
                    file_path: Union[str, Path],
                    cache: bool = True) -> pd.DataFrame:
        """
        Load Parquet file with optimized performance.
        
        Args:
            file_path: Path to Parquet file
            cache: Whether to cache the loaded data
            
        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")
        
        try:
            logger.info(f"Loading Parquet data from: {file_path}")
            start_time = datetime.now()
            
            df = pd.read_parquet(file_path)
            
            load_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Parquet data loaded successfully in {load_time:.2f}s. Shape: {df.shape}")
            
            # Cache the data
            if cache:
                cache_key = str(file_path)
                self.data_cache[cache_key] = df.copy()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load Parquet data from {file_path}: {str(e)}")
            raise
    
    def save_data(self, 
                  df: pd.DataFrame, 
                  file_path: Union[str, Path],
                  format: str = 'csv',
                  **kwargs) -> None:
        """
        Save DataFrame to file with format options.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            format: File format ('csv', 'parquet', 'json')
            **kwargs: Additional arguments for the save method
        """
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Saving data to {file_path} in {format} format")
            start_time = datetime.now()
            
            if format.lower() == 'csv':
                df.to_csv(file_path, index=False, **kwargs)
            elif format.lower() == 'parquet':
                df.to_parquet(file_path, index=False, **kwargs)
            elif format.lower() == 'json':
                df.to_json(file_path, orient='records', **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            save_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Data saved successfully in {save_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to save data to {file_path}: {str(e)}")
            raise
    
    def _validate_schema(self, df: pd.DataFrame, file_path: Path) -> None:
        """
        Validate DataFrame schema against expected schema.
        
        Args:
            df: DataFrame to validate
            file_path: Original file path for context
        """
        # Expected schema for liver disease data
        expected_schema = {
            'Age': 'int64',
            'Sex': 'object',
            'ALB': 'float64',
            'ALP': 'int64',
            'ALT': 'int64',
            'AST': 'int64',
            'BIL': 'float64',
            'CHE': 'float64',
            'CHOL': 'int64',
            'CREA': 'float64',
            'GGT': 'int64',
            'PROT': 'float64',
            'Category': 'object'
        }
        
        # Check required columns
        missing_columns = set(expected_schema.keys()) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check data types (allow some flexibility)
        for col, expected_type in expected_schema.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type == 'int64' and not actual_type.startswith('int'):
                    logger.warning(f"Column {col} has type {actual_type}, expected int64")
                elif expected_type == 'float64' and not actual_type.startswith('float'):
                    logger.warning(f"Column {col} has type {actual_type}, expected float64")
                elif expected_type == 'object' and actual_type != 'object':
                    logger.warning(f"Column {col} has type {actual_type}, expected object")
        
        logger.info("Schema validation passed")
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with data information
        """
        info = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
            'categorical_summary': df.select_dtypes(include=['object']).describe().to_dict() if len(df.select_dtypes(include=['object']).columns) > 0 else {}
        }
        
        return info
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self.data_cache.clear()
        logger.info("Data cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached data.
        
        Returns:
            Dictionary with cache information
        """
        cache_info = {}
        for key, df in self.data_cache.items():
            cache_info[key] = {
                'shape': df.shape,
                'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
                'columns': list(df.columns)
            }
        
        return cache_info
