"""
Production-grade model trainer with MLflow integration and comprehensive tracking
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix)
import mlflow
import mlflow.sklearn
import optuna
from typing import Dict, List, Any, Optional, Tuple
import logging
import joblib
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Production-grade model trainer with comprehensive MLflow integration.
    
    Features:
    - Multiple algorithm support
    - Hyperparameter optimization with Optuna
    - MLflow experiment tracking
    - Cross-validation with stratified splits
    - Model evaluation and comparison
    - Model persistence and versioning
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelTrainer with configuration.
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.mlops_config = config.get('mlops', {})
        
        # Initialize MLflow
        self._setup_mlflow()
        
        # Model registry
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.training_history = []
        
        logger.info("ModelTrainer initialized")
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        try:
            tracking_uri = self.mlops_config.get('experiment_tracking', {}).get('tracking_uri')
            experiment_name = self.mlops_config.get('experiment_tracking', {}).get('experiment_name')
            
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            
            if experiment_name:
                try:
                    mlflow.create_experiment(experiment_name)
                except:
                    pass  # Experiment already exists
                mlflow.set_experiment(experiment_name)
            
            logger.info(f"MLflow setup complete. Experiment: {experiment_name}")
            
        except Exception as e:
            logger.warning(f"MLflow setup failed: {str(e)}. Continuing without MLflow.")
    
    def train_models(self, 
                    X_train: pd.DataFrame, 
                    y_train: pd.Series,
                    X_val: Optional[pd.DataFrame] = None,
                    y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train multiple models with hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training with hyperparameter optimization")
        
        # Get model algorithms from config
        algorithms = self.model_config.get('algorithms', ['random_forest'])
        
        results = {}
        
        for algorithm in algorithms:
            logger.info(f"Training {algorithm} model...")
            
            try:
                with mlflow.start_run(run_name=f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                    # Train model
                    model, metrics = self._train_single_model(
                        algorithm, X_train, y_train, X_val, y_val
                    )
                    
                    # Store model and results
                    self.models[algorithm] = {
                        'model': model,
                        'metrics': metrics,
                        'algorithm': algorithm
                    }
                    
                    results[algorithm] = metrics
                    
                    # Update best model
                    if metrics['cv_score_mean'] > self.best_score:
                        self.best_score = metrics['cv_score_mean']
                        self.best_model = model
                        self.best_algorithm = algorithm
                    
                    # Log to MLflow
                    self._log_to_mlflow(algorithm, model, metrics, X_train, y_train)
                    
                    logger.info(f"{algorithm} training completed. CV Score: {metrics['cv_score_mean']:.4f}")
                    
            except Exception as e:
                logger.error(f"Failed to train {algorithm}: {str(e)}")
                results[algorithm] = {'error': str(e)}
        
        # Generate training summary
        training_summary = self._generate_training_summary(results)
        
        logger.info(f"Model training completed. Best model: {self.best_algorithm} "
                   f"with CV score: {self.best_score:.4f}")
        
        return training_summary
    
    def _train_single_model(self, 
                           algorithm: str,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           X_val: Optional[pd.DataFrame] = None,
                           y_val: Optional[pd.Series] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a single model with hyperparameter optimization.
        
        Args:
            algorithm: Model algorithm name
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Tuple of (trained_model, metrics)
        """
        start_time = datetime.now()
        
        # Get hyperparameters from config
        hyperparams = self.model_config.get('hyperparameters', {}).get(algorithm, {})
        
        if not hyperparams:
            # Use default hyperparameters
            model = self._get_default_model(algorithm)
            model.fit(X_train, y_train)
        else:
            # Hyperparameter optimization with Optuna
            model = self._optimize_hyperparameters(algorithm, hyperparams, X_train, y_train)
        
        # Cross-validation
        cv_scores = self._cross_validate(model, X_train, y_train)
        
        # Validation evaluation (if validation data provided)
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_predictions = model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_predictions, prefix='val_')
        
        # Training metrics
        train_predictions = model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_predictions, prefix='train_')
        
        # Combine all metrics
        metrics = {
            'cv_score_mean': cv_scores.mean(),
            'cv_score_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'training_time': (datetime.now() - start_time).total_seconds(),
            **train_metrics,
            **val_metrics
        }
        
        return model, metrics
    
    def _get_default_model(self, algorithm: str):
        """Get default model instance for algorithm."""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.model_config.get('random_state', 42),
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.model_config.get('random_state', 42)
            ),
            'svm': SVC(
                random_state=self.model_config.get('random_state', 42),
                probability=True
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.model_config.get('random_state', 42),
                max_iter=1000
            ),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'lda': LinearDiscriminantAnalysis()
        }
        
        if algorithm not in models:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        return models[algorithm]
    
    def _optimize_hyperparameters(self, 
                                 algorithm: str,
                                 hyperparams: Dict[str, List],
                                 X_train: pd.DataFrame,
                                 y_train: pd.Series) -> Any:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            algorithm: Model algorithm name
            hyperparams: Hyperparameter search space
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Optimized model
        """
        def objective(trial):
            # Suggest hyperparameters
            params = {}
            for param_name, param_values in hyperparams.items():
                if isinstance(param_values[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_values[0], param_values[-1])
                elif isinstance(param_values[0], float):
                    params[param_name] = trial.suggest_float(param_name, param_values[0], param_values[-1])
                elif isinstance(param_values[0], str):
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
            
            # Create and train model
            model = self._get_default_model(algorithm)
            model.set_params(**params)
            
            # Cross-validation score
            cv_scores = self._cross_validate(model, X_train, y_train)
            return cv_scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)  # Configurable
        
        # Get best parameters
        best_params = study.best_params
        logger.info(f"Best parameters for {algorithm}: {best_params}")
        
        # Train final model with best parameters
        model = self._get_default_model(algorithm)
        model.set_params(**best_params)
        model.fit(X_train, y_train)
        
        return model
    
    def _cross_validate(self, model: Any, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Perform cross-validation."""
        cv_folds = self.model_config.get('cv_folds', 5)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                           random_state=self.model_config.get('random_state', 42))
        
        cv_scores = cross_val_score(
            model, X, y, cv=cv, scoring='accuracy', n_jobs=-1
        )
        
        return cv_scores
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, prefix: str = '') -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        metrics = {}
        
        metrics[f'{prefix}accuracy'] = accuracy_score(y_true, y_pred)
        metrics[f'{prefix}precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics[f'{prefix}recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics[f'{prefix}f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        return metrics
    
    def _log_to_mlflow(self, 
                      algorithm: str, 
                      model: Any, 
                      metrics: Dict[str, Any],
                      X_train: pd.DataFrame,
                      y_train: pd.Series):
        """Log model and metrics to MLflow."""
        try:
            # Log parameters
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=f"{algorithm}_liver_disease"
            )
            
            # Log training data info
            mlflow.log_metric("training_samples", len(X_train))
            mlflow.log_metric("training_features", len(X_train.columns))
            
            # Log confusion matrix as artifact
            train_pred = model.predict(X_train)
            cm = confusion_matrix(y_train, train_pred)
            cm_df = pd.DataFrame(cm, 
                               index=y_train.unique(), 
                               columns=y_train.unique())
            cm_df.to_csv("confusion_matrix.csv")
            mlflow.log_artifact("confusion_matrix.csv")
            
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {str(e)}")
    
    def _generate_training_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive training summary."""
        summary = {
            'training_timestamp': datetime.now().isoformat(),
            'total_models_trained': len([r for r in results.values() if 'error' not in r]),
            'failed_models': len([r for r in results.values() if 'error' in r]),
            'best_model': self.best_algorithm,
            'best_score': self.best_score,
            'model_results': results,
            'training_config': self.model_config
        }
        
        return summary
    
    def save_model(self, model_name: str, save_path: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            model_name: Name of the model to save
            save_path: Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_data = self.models[model_name]
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare model data for saving
        model_package = {
            'model': model_data['model'],
            'metadata': {
                'algorithm': model_data['algorithm'],
                'metrics': model_data['metrics'],
                'trained_at': datetime.now().isoformat(),
                'training_config': self.model_config,
                'feature_names': list(model_data['model'].feature_names_in_) if hasattr(model_data['model'], 'feature_names_in_') else None
            }
        }
        
        # Save model
        joblib.dump(model_package, save_path)
        
        logger.info(f"Model {model_name} saved to {save_path}")
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_package = joblib.load(model_path)
        model = model_package['model']
        
        logger.info(f"Model loaded from {model_path}")
        
        return model
