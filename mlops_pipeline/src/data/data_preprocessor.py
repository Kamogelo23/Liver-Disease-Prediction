"""
Production-grade data preprocessor with comprehensive transformations
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from typing import Dict, List, Any, Optional, Tuple
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Comprehensive data preprocessor for production ML pipelines.
    
    Features:
    - Missing value imputation
    - Feature scaling/normalization
    - Encoding categorical variables
    - Outlier handling
    - Feature transformation
    - State persistence for inference
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataPreprocessor with configuration.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config
        self.feature_config = config.get('features', {})
        self.is_fitted = False
        
        # Initialize transformers
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}
        self.feature_names = []
        
        logger.info("DataPreprocessor initialized")
    
    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            df: Training DataFrame
            target_col: Name of target column (optional)
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting preprocessor on training data...")
        
        try:
            # Separate features and target
            if target_col and target_col in df.columns:
                X = df.drop(columns=[target_col])
                self.target_col = target_col
            else:
                X = df.copy()
                self.target_col = None
            
            # Store feature names
            self.feature_names = list(X.columns)
            
            # Identify feature types
            self.categorical_features = self.feature_config.get('categorical_features', [])
            self.numerical_features = self.feature_config.get('numerical_features', [])
            
            # Fit imputers
            self._fit_imputers(X)
            
            # Fit encoders for categorical features
            self._fit_encoders(X)
            
            # Fit scalers for numerical features
            self._fit_scalers(X)
            
            self.is_fitted = True
            logger.info("Preprocessor fitted successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"Failed to fit preprocessor: {str(e)}")
            raise
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        logger.info("Transforming data...")
        
        try:
            X = df.copy()
            
            # Handle missing values
            X = self._transform_imputation(X)
            
            # Encode categorical features
            X = self._transform_encoding(X)
            
            # Scale numerical features
            X = self._transform_scaling(X)
            
            # Handle outliers if configured
            if self.config.get('data_quality', {}).get('outlier_detection', {}).get('method'):
                X = self._handle_outliers(X)
            
            logger.info(f"Data transformed successfully. Shape: {X.shape}")
            
            return X
            
        except Exception as e:
            logger.error(f"Failed to transform data: {str(e)}")
            raise
    
    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Fit and transform data in one step.
        
        Args:
            df: DataFrame to fit and transform
            target_col: Name of target column (optional)
            
        Returns:
            Transformed DataFrame
        """
        self.fit(df, target_col)
        
        # Transform features
        if target_col and target_col in df.columns:
            X_transformed = self.transform(df.drop(columns=[target_col]))
            X_transformed[target_col] = df[target_col]
            return X_transformed
        else:
            return self.transform(df)
    
    def _fit_imputers(self, X: pd.DataFrame) -> None:
        """Fit imputers for missing value handling."""
        logger.info("Fitting imputers...")
        
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if col in self.categorical_features:
                    # Use most frequent for categorical
                    imputer = SimpleImputer(strategy='most_frequent')
                elif col in self.numerical_features:
                    # Use KNN imputer for numerical
                    imputer = KNNImputer(n_neighbors=5)
                else:
                    # Default to median
                    imputer = SimpleImputer(strategy='median')
                
                imputer.fit(X[[col]])
                self.imputers[col] = imputer
                logger.debug(f"Fitted imputer for column: {col}")
    
    def _transform_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted imputers."""
        X = X.copy()
        
        for col, imputer in self.imputers.items():
            if col in X.columns:
                X[col] = imputer.transform(X[[col]])
        
        return X
    
    def _fit_encoders(self, X: pd.DataFrame) -> None:
        """Fit encoders for categorical variables."""
        logger.info("Fitting encoders...")
        
        for col in self.categorical_features:
            if col in X.columns:
                encoder = LabelEncoder()
                # Handle NaN values before encoding
                mask = X[col].notna()
                if mask.sum() > 0:
                    encoder.fit(X.loc[mask, col])
                    self.encoders[col] = encoder
                    logger.debug(f"Fitted encoder for column: {col}")
    
    def _transform_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical variables using fitted encoders."""
        X = X.copy()
        
        for col, encoder in self.encoders.items():
            if col in X.columns:
                mask = X[col].notna()
                if mask.sum() > 0:
                    try:
                        X.loc[mask, col] = encoder.transform(X.loc[mask, col])
                    except ValueError as e:
                        logger.warning(f"Unknown categories in column {col}, using most frequent")
                        # Handle unknown categories
                        known_classes = set(encoder.classes_)
                        X.loc[mask, col] = X.loc[mask, col].apply(
                            lambda x: x if x in known_classes else encoder.classes_[0]
                        )
                        X.loc[mask, col] = encoder.transform(X.loc[mask, col])
        
        return X
    
    def _fit_scalers(self, X: pd.DataFrame) -> None:
        """Fit scalers for numerical features."""
        logger.info("Fitting scalers...")
        
        scaling_method = self.feature_config.get('scaling', {}).get('method', 'standard')
        
        for col in self.numerical_features:
            if col in X.columns:
                if scaling_method == 'standard':
                    scaler = StandardScaler()
                elif scaling_method == 'robust':
                    scaler = RobustScaler()
                elif scaling_method == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()
                
                # Fit on non-null values
                mask = X[col].notna()
                if mask.sum() > 0:
                    scaler.fit(X.loc[mask, [col]])
                    self.scalers[col] = scaler
                    logger.debug(f"Fitted {scaling_method} scaler for column: {col}")
    
    def _transform_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform numerical features using fitted scalers."""
        X = X.copy()
        
        for col, scaler in self.scalers.items():
            if col in X.columns:
                mask = X[col].notna()
                if mask.sum() > 0:
                    X.loc[mask, col] = scaler.transform(X.loc[mask, [col]])
        
        return X
    
    def _handle_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numerical features."""
        X = X.copy()
        
        outlier_config = self.config.get('data_quality', {}).get('outlier_detection', {})
        method = outlier_config.get('method', 'clip')
        
        for col in self.numerical_features:
            if col in X.columns:
                if method == 'clip':
                    # Clip outliers to percentiles
                    lower = X[col].quantile(0.01)
                    upper = X[col].quantile(0.99)
                    X[col] = X[col].clip(lower, upper)
                elif method == 'remove':
                    # Mark outliers for removal (handled by caller)
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    X[col] = X[col].mask((X[col] < lower_bound) | (X[col] > upper_bound))
        
        return X
    
    def save(self, path: str) -> None:
        """
        Save fitted preprocessor to disk.
        
        Args:
            path: Path to save preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before saving")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'scalers': self.scalers,
            'imputers': self.imputers,
            'encoders': self.encoders,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'target_col': self.target_col,
            'is_fitted': self.is_fitted,
            'config': self.config
        }, path)
        
        logger.info(f"Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'DataPreprocessor':
        """
        Load fitted preprocessor from disk.
        
        Args:
            path: Path to load preprocessor from
            
        Returns:
            Loaded DataPreprocessor instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {path}")
        
        state = joblib.load(path)
        
        preprocessor = cls(state['config'])
        preprocessor.scalers = state['scalers']
        preprocessor.imputers = state['imputers']
        preprocessor.encoders = state['encoders']
        preprocessor.feature_names = state['feature_names']
        preprocessor.categorical_features = state['categorical_features']
        preprocessor.numerical_features = state['numerical_features']
        preprocessor.target_col = state['target_col']
        preprocessor.is_fitted = state['is_fitted']
        
        logger.info(f"Preprocessor loaded from {path}")
        
        return preprocessor
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names after preprocessing."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        return self.feature_names
