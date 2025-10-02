#!/usr/bin/env python3
"""
=============================================================================
Liver Disease Prediction - Professional Python Script
=============================================================================
Author: Improved version of original project
Date: 2024
Description: Comprehensive analysis and prediction of liver disease using HCV data
             with modern Python ML libraries and best practices
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import logging.config
from datetime import datetime
from typing import Tuple, Dict, List, Optional
import joblib
import json
import time
import functools

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Visualization Libraries
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Setup comprehensive logging configuration with multiple handlers and structured output.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str): Custom log file path
        
    Returns:
        logging.Logger: Configured logger instance
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"output/liver_disease_prediction_{timestamp}.log"
    
    # Create output directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True)
    
    # Define logging configuration
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s:%(lineno)-4d | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(asctime)s | %(levelname)-8s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s"}',
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": str(log_file),
                "mode": "w",
                "encoding": "utf-8"
            },
            "json_file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "json",
                "filename": str(log_file).replace('.log', '_structured.json'),
                "mode": "w",
                "encoding": "utf-8"
            }
        },
        "loggers": {
            "": {  # Root logger
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False
            },
            "structured": {  # Structured logger for metrics
                "level": "DEBUG",
                "handlers": ["json_file"],
                "propagate": False
            }
        }
    }
    
    # Apply logging configuration
    logging.config.dictConfig(logging_config)
    
    # Create logger
    logger = logging.getLogger(__name__)
    structured_logger = logging.getLogger("structured")
    
    # Log startup information
    logger.info("=" * 80)
    logger.info("LIVER DISEASE PREDICTION PIPELINE STARTED")
    logger.info("=" * 80)
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    return logger

def log_performance(func):
    """
    Decorator to log function execution time and performance metrics.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        logger.info(f"Starting {func.__name__}...")
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed {func.__name__} in {execution_time:.2f} seconds")
            
            # Log to structured logger for performance tracking
            structured_logger = logging.getLogger("structured")
            structured_logger.info(json.dumps({
                "event": "function_completed",
                "function_name": func.__name__,
                "execution_time_seconds": round(execution_time, 2),
                "status": "success"
            }))
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {execution_time:.2f} seconds: {str(e)}")
            
            # Log error to structured logger
            structured_logger = logging.getLogger("structured")
            structured_logger.error(json.dumps({
                "event": "function_failed",
                "function_name": func.__name__,
                "execution_time_seconds": round(execution_time, 2),
                "error": str(e),
                "status": "error"
            }))
            
            raise
    
    return wrapper

def log_data_info(data: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Log comprehensive information about a DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame to analyze
        name (str): Name for the dataset
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"=== {name.upper()} INFORMATION ===")
    logger.info(f"Shape: {data.shape}")
    logger.info(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types
    logger.info("Data types:")
    for dtype, count in data.dtypes.value_counts().items():
        logger.info(f"  {dtype}: {count} columns")
    
    # Missing values
    missing_values = data.isnull().sum()
    total_missing = missing_values.sum()
    if total_missing > 0:
        logger.warning(f"Missing values: {total_missing} ({total_missing/data.size*100:.2f}%)")
        for col, missing in missing_values[missing_values > 0].items():
            logger.warning(f"  {col}: {missing} ({missing/len(data)*100:.2f}%)")
    else:
        logger.info("No missing values found")
    
    # Categorical variables
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        logger.info(f"Categorical variables: {list(categorical_cols)}")
        for col in categorical_cols:
            unique_count = data[col].nunique()
            logger.info(f"  {col}: {unique_count} unique values")
    
    # Numeric variables summary
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        logger.info(f"Numeric variables: {list(numeric_cols)}")
        summary_stats = data[numeric_cols].describe()
        logger.info("Summary statistics:")
        for col in numeric_cols:
            logger.info(f"  {col}: mean={summary_stats.loc['mean', col]:.3f}, "
                       f"std={summary_stats.loc['std', col]:.3f}, "
                       f"min={summary_stats.loc['min', col]:.3f}, "
                       f"max={summary_stats.loc['max', col]:.3f}")

def log_model_performance(model_name: str, metrics: Dict, execution_time: float) -> None:
    """
    Log model performance metrics in a structured format.
    
    Args:
        model_name (str): Name of the model
        metrics (Dict): Performance metrics
        execution_time (float): Training time in seconds
    """
    logger = logging.getLogger(__name__)
    structured_logger = logging.getLogger("structured")
    
    # Log to main logger
    logger.info(f"=== {model_name.upper()} PERFORMANCE ===")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    logger.info(f"Training time: {execution_time:.2f} seconds")
    
    # Log to structured logger
    structured_logger.info(json.dumps({
        "event": "model_performance",
        "model_name": model_name,
        "metrics": metrics,
        "training_time_seconds": round(execution_time, 2),
        "timestamp": datetime.now().isoformat()
    }))

# Setup logging
logger = setup_logging()

# Suppress warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

class LiverDiseasePredictor:
    """
    A comprehensive class for liver disease prediction using machine learning.
    
    This class handles data loading, preprocessing, model training, evaluation,
    and visualization for liver disease prediction tasks.
    """
    
    def __init__(self, data_path: str = "hcvdat0.csv"):
        """
        Initialize the Liver Disease Predictor.
        
        Args:
            data_path (str): Path to the dataset file
        """
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.results = {}
        
        # Create output directory
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize performance tracking
        self.performance_metrics = {
            "data_loading_time": 0,
            "preprocessing_time": 0,
            "training_time": 0,
            "evaluation_time": 0,
            "total_time": 0
        }
        
        logger.info("Liver Disease Predictor initialized")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def create_sample_dataset(self, n_samples: int = 615) -> pd.DataFrame:
        """
        Create a sample dataset for demonstration purposes.
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Sample dataset with HCV-like features
        """
        np.random.seed(123)
        
        # Define disease categories and their probabilities
        categories = ["Blood donor", "Cirrhosis", "Fibrosis", "Hepatitis"]
        category_probs = [0.4, 0.2, 0.2, 0.2]
        
        # Generate synthetic data based on medical literature
        data = pd.DataFrame({
            'Age': np.random.normal(45, 15, n_samples).astype(int),
            'Sex': np.random.choice(['m', 'f'], n_samples),
            'ALB': np.random.normal(4.2, 0.6, n_samples).round(2),  # Albumin
            'ALP': np.random.normal(95, 35, n_samples).round(0),    # Alkaline Phosphatase
            'ALT': np.random.normal(35, 25, n_samples).round(0),    # Alanine Aminotransferase
            'AST': np.random.normal(30, 20, n_samples).round(0),    # Aspartate Aminotransferase
            'BIL': np.random.normal(0.8, 0.4, n_samples).round(2),  # Bilirubin
            'CHE': np.random.normal(8.5, 2.0, n_samples).round(2),  # Cholinesterase
            'CHOL': np.random.normal(200, 50, n_samples).round(0),  # Cholesterol
            'CREA': np.random.normal(1.0, 0.3, n_samples).round(2), # Creatinine
            'GGT': np.random.normal(40, 30, n_samples).round(0),    # Gamma-glutamyl transferase
            'PROT': np.random.normal(7.0, 0.8, n_samples).round(2), # Protein
            'Category': np.random.choice(categories, n_samples, p=category_probs)
        })
        
        # Add some missing values to make it realistic
        missing_indices = np.random.choice(n_samples, 31, replace=False)
        data.loc[missing_indices[:10], 'ALB'] = np.nan
        data.loc[missing_indices[10:20], 'ALP'] = np.nan
        data.loc[missing_indices[20:31], 'BIL'] = np.nan
        
        logger.info(f"Sample dataset created with {n_samples} observations and {len(data.columns)} variables")
        return data
    
    @log_performance
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset with error handling.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        start_time = time.time()
        
        if not os.path.exists(self.data_path):
            logger.warning(f"Dataset file {self.data_path} not found. Creating sample dataset...")
            self.data = self.create_sample_dataset()
        else:
            try:
                logger.info(f"Loading dataset from {self.data_path}")
                self.data = pd.read_csv(self.data_path)
                logger.info(f"Dataset loaded successfully! Shape: {self.data.shape}")
            except Exception as e:
                logger.error(f"Error loading data: {str(e)}")
                logger.info("Creating sample dataset instead...")
                self.data = self.create_sample_dataset()
        
        # Log comprehensive data information
        log_data_info(self.data, "Original Dataset")
        
        # Update performance metrics
        self.performance_metrics["data_loading_time"] = time.time() - start_time
        
        return self.data
    
    def explore_data(self) -> None:
        """
        Perform comprehensive exploratory data analysis.
        """
        logger.info("Starting exploratory data analysis...")
        
        # Basic information
        logger.info(f"Dataset shape: {self.data.shape}")
        logger.info(f"Columns: {list(self.data.columns)}")
        
        # Data types
        logger.info("Data types:")
        logger.info(self.data.dtypes)
        
        # Missing values
        missing_values = self.data.isnull().sum()
        logger.info("Missing values per column:")
        logger.info(missing_values[missing_values > 0])
        
        # Statistical summary
        logger.info("Statistical summary:")
        logger.info(self.data.describe())
        
        # Create visualizations
        self._create_eda_plots()
    
    def _create_eda_plots(self) -> None:
        """
        Create exploratory data analysis plots.
        """
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Category distribution
        category_counts = self.data['Category'].value_counts()
        axes[0, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Distribution of Disease Categories')
        
        # 2. Age distribution by category
        sns.boxplot(data=self.data, x='Category', y='Age', ax=axes[0, 1])
        axes[0, 1].set_title('Age Distribution by Category')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Correlation heatmap
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
        axes[1, 0].set_title('Feature Correlation Matrix')
        
        # 4. Missing values pattern
        missing_data = self.data.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            axes[1, 1].bar(missing_data.index, missing_data.values)
            axes[1, 1].set_title('Missing Values by Feature')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                           transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title('Missing Values by Feature')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create additional detailed plots
        self._create_detailed_plots()
    
    def _create_detailed_plots(self) -> None:
        """
        Create detailed plots for key features.
        """
        # Feature distributions by category
        key_features = ['ALB', 'ALT', 'AST', 'BIL', 'ALP']
        available_features = [col for col in key_features if col in self.data.columns]
        
        if len(available_features) > 0:
            n_features = len(available_features)
            n_cols = 2
            n_rows = (n_features + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, feature in enumerate(available_features):
                row, col = i // n_cols, i % n_cols
                sns.boxplot(data=self.data, x='Category', y=feature, ax=axes[row, col])
                axes[row, col].set_title(f'{feature} Distribution by Category')
                axes[row, col].tick_params(axis='x', rotation=45)
            
            # Hide empty subplots
            for i in range(len(available_features), n_rows * n_cols):
                row, col = i // n_cols, i % n_cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the data for machine learning.
        
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        logger.info("Starting data preprocessing...")
        
        # Create a copy of the data
        df = self.data.copy()
        
        # Handle categorical variables
        df['Sex'] = df['Sex'].map({'m': 1, 'f': 0})
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['Category']]
        X = df[feature_cols]
        y = df['Category']
        
        # Handle missing values using KNN imputation
        if X.isnull().sum().sum() > 0:
            logger.info("Handling missing values using KNN imputation...")
            imputer = KNNImputer(n_neighbors=5)
            X_imputed = imputer.fit_transform(X)
            X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Combine features and target
        processed_df = X.copy()
        processed_df['Category'] = y_encoded
        
        self.processed_data = processed_df
        logger.info(f"Data preprocessing completed. Final shape: {processed_df.shape}")
        
        return processed_df
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split data into training and testing sets.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random state for reproducibility
            
        Returns:
            Tuple: Training and testing sets
        """
        logger.info("Splitting data into training and testing sets...")
        
        X = self.processed_data.drop('Category', axis=1)
        y = self.processed_data['Category']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        logger.info(f"Training set shape: {self.X_train.shape}")
        logger.info(f"Test set shape: {self.X_test.shape}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def handle_class_imbalance(self) -> Tuple:
        """
        Handle class imbalance using SMOTE.
        
        Returns:
            Tuple: Balanced training sets
        """
        logger.info("Handling class imbalance using SMOTE...")
        
        try:
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(self.X_train_scaled, self.y_train)
            
            logger.info(f"Original training set shape: {self.X_train_scaled.shape}")
            logger.info(f"Balanced training set shape: {X_train_balanced.shape}")
            
            # Check class distribution
            unique, counts = np.unique(y_train_balanced, return_counts=True)
            logger.info("Balanced class distribution:")
            for cls, count in zip(unique, counts):
                class_name = self.label_encoder.inverse_transform([cls])[0]
                logger.info(f"  {class_name}: {count}")
            
            return X_train_balanced, y_train_balanced
        except Exception as e:
            logger.warning(f"SMOTE failed: {str(e)}. Using original data.")
            return self.X_train_scaled, self.y_train
    
    @log_performance
    def train_models(self) -> Dict:
        """
        Train multiple machine learning models.
        
        Returns:
            Dict: Trained models
        """
        start_time = time.time()
        logger.info("Starting model training process...")
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_class_imbalance()
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            'SVM': SVC(
                kernel='rbf', random_state=42, probability=True,
                class_weight='balanced'
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42, class_weight='balanced'
            ),
            'k-NN': KNeighborsClassifier(
                n_neighbors=5
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced'
            ),
            'LDA': LinearDiscriminantAnalysis()
        }
        
        # Train models
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model_start_time = time.time()
            
            try:
                model.fit(X_train_balanced, y_train_balanced)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5)
                model_time = time.time() - model_start_time
                
                # Log model performance
                model_metrics = {
                    'cv_score_mean': cv_scores.mean(),
                    'cv_score_std': cv_scores.std(),
                    'training_time': model_time
                }
                log_model_performance(name, model_metrics, model_time)
                
                logger.info(f"{name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                self.models[name] = model
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
        
        # Find best model
        self._evaluate_models()
        
        # Update performance metrics
        self.performance_metrics["training_time"] = time.time() - start_time
        
        logger.info(f"Model training completed in {self.performance_metrics['training_time']:.2f} seconds")
        return self.models
    
    def _evaluate_models(self) -> None:
        """
        Evaluate all trained models and select the best one.
        """
        logger.info("Evaluating models...")
        
        best_score = 0
        best_model_name = None
        
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            
            if accuracy > best_score:
                best_score = accuracy
                best_model_name = name
        
        self.best_model = self.models[best_model_name]
        logger.info(f"Best model: {best_model_name} with accuracy: {best_score:.4f}")
    
    def evaluate_model(self, model_name: str = None) -> Dict:
        """
        Evaluate a specific model or the best model.
        
        Args:
            model_name (str): Name of the model to evaluate
            
        Returns:
            Dict: Evaluation metrics
        """
        if model_name is None:
            model = self.best_model
            model_name = "Best Model"
        else:
            model = self.models[model_name]
        
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(self.X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_pred, average='weighted'),
            'f1': f1_score(self.y_test, y_pred, average='weighted')
        }
        
        # Print results
        logger.info(f"\n{model_name} Evaluation Results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1']:.4f}")
        
        # Classification report
        class_names = self.label_encoder.classes_
        logger.info("\nClassification Report:")
        logger.info(classification_report(self.y_test, y_pred, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def create_feature_importance_plot(self) -> None:
        """
        Create feature importance plot for tree-based models.
        """
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = self.processed_data.drop('Category', axis=1).columns
            importances = self.best_model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 8))
            plt.title('Feature Importance')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            logger.info("Feature importance not available for this model type.")
    
    def save_model(self, filename: str = "best_liver_disease_model.pkl") -> None:
        """
        Save the best model to disk.
        
        Args:
            filename (str): Filename for the saved model
        """
        if self.best_model is not None:
            model_path = self.output_dir / filename
            joblib.dump(self.best_model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Also save scaler and label encoder
            joblib.dump(self.scaler, self.output_dir / "scaler.pkl")
            joblib.dump(self.label_encoder, self.output_dir / "label_encoder.pkl")
        else:
            logger.warning("No model to save. Train models first.")
    
    def predict_new_sample(self, sample_data: Dict) -> str:
        """
        Predict liver disease for a new sample.
        
        Args:
            sample_data (Dict): Dictionary containing feature values
            
        Returns:
            str: Predicted category
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Train models first.")
        
        # Convert to DataFrame
        sample_df = pd.DataFrame([sample_data])
        
        # Preprocess (same as training data)
        if 'Sex' in sample_df.columns:
            sample_df['Sex'] = sample_df['Sex'].map({'m': 1, 'f': 0})
        
        # Scale features
        sample_scaled = self.scaler.transform(sample_df)
        
        # Make prediction
        prediction = self.best_model.predict(sample_scaled)[0]
        predicted_category = self.label_encoder.inverse_transform([prediction])[0]
        
        return predicted_category

def main():
    """
    Main function to run the liver disease prediction pipeline.
    """
    pipeline_start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("STARTING LIVER DISEASE PREDICTION PIPELINE")
    logger.info("=" * 80)
    
    try:
        # Initialize predictor
        logger.info("Initializing Liver Disease Predictor...")
        predictor = LiverDiseasePredictor()
        
        # Load data
        logger.info("Step 1: Loading data...")
        predictor.load_data()
        
        # Explore data
        logger.info("Step 2: Performing exploratory data analysis...")
        predictor.explore_data()
        
        # Preprocess data
        logger.info("Step 3: Preprocessing data...")
        predictor.preprocess_data()
        
        # Split data
        logger.info("Step 4: Splitting data into train/test sets...")
        predictor.split_data()
        
        # Train models
        logger.info("Step 5: Training machine learning models...")
        predictor.train_models()
        
        # Evaluate best model
        logger.info("Step 6: Evaluating model performance...")
        predictor.evaluate_model()
        
        # Create feature importance plot
        logger.info("Step 7: Creating visualizations...")
        predictor.create_feature_importance_plot()
        
        # Save model
        logger.info("Step 8: Saving model and results...")
        predictor.save_model()
        
        # Example prediction
        logger.info("Step 9: Testing prediction functionality...")
        sample_data = {
            'Age': 45,
            'Sex': 'm',
            'ALB': 4.0,
            'ALP': 100,
            'ALT': 40,
            'AST': 35,
            'BIL': 1.0,
        'CHE': 8.0,
            'CHOL': 200,
            'CREA': 1.1,
            'GGT': 45,
            'PROT': 7.2
        }
        
        try:
            prediction = predictor.predict_new_sample(sample_data)
            logger.info(f"Sample prediction successful: {prediction}")
        except Exception as e:
            logger.error(f"Sample prediction failed: {str(e)}")
        
        # Calculate total pipeline time
        total_time = time.time() - pipeline_start_time
        predictor.performance_metrics["total_time"] = total_time
        
        # Log final summary
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("PERFORMANCE SUMMARY:")
        for metric, time_taken in predictor.performance_metrics.items():
            logger.info(f"  {metric}: {time_taken:.2f} seconds")
        logger.info(f"Total pipeline time: {total_time:.2f} seconds")
        logger.info(f"Results saved in '{predictor.output_dir}' directory")
        
        # Log structured summary
        structured_logger = logging.getLogger("structured")
        structured_logger.info(json.dumps({
            "event": "pipeline_completed",
            "status": "success",
            "total_execution_time": round(total_time, 2),
            "performance_metrics": predictor.performance_metrics,
            "timestamp": datetime.now().isoformat()
        }))
        
    except Exception as e:
        total_time = time.time() - pipeline_start_time
        logger.error("=" * 80)
        logger.error("PIPELINE FAILED!")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}")
        logger.error(f"Failed after {total_time:.2f} seconds")
        
        # Log structured error
        structured_logger = logging.getLogger("structured")
        structured_logger.error(json.dumps({
            "event": "pipeline_failed",
            "status": "error",
            "error": str(e),
            "execution_time": round(total_time, 2),
            "timestamp": datetime.now().isoformat()
        }))
        
        raise

if __name__ == "__main__":
    main()
