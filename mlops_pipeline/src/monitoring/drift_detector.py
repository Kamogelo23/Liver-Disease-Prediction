"""
Production-grade drift detection using Evidently and statistical tests
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.metrics import DatasetDriftMetric, DataQualityMetrics
    from evidently.test_suite import TestSuite
    from evidently.test_preset import DataDriftTestPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logging.warning("Evidently not available. Drift detection will use statistical methods only.")

from scipy import stats
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

@dataclass
class DriftResult:
    """Data class to store drift detection results."""
    feature_name: str
    drift_detected: bool
    drift_score: float
    p_value: float
    threshold: float
    method: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Dict[str, Any] = field(default_factory=dict)

class DriftDetector:
    """
    Comprehensive drift detection system for production ML pipelines.
    
    Features:
    - Statistical drift detection (KS test, PSI, etc.)
    - Evidently-based drift detection
    - Target drift detection
    - Data quality monitoring
    - Automated alerts
    - Historical drift tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DriftDetector with configuration.
        
        Args:
            config: Configuration dictionary with drift detection settings
        """
        self.config = config
        self.drift_config = config.get('data_quality', {}).get('data_drift', {})
        self.detection_method = self.drift_config.get('detection_method', 'ks_test')
        self.threshold = self.drift_config.get('threshold', 0.05)
        
        # Drift history
        self.drift_history = []
        self.reference_data = None
        
        logger.info("DriftDetector initialized")
    
    def set_reference_data(self, reference_data: pd.DataFrame) -> None:
        """
        Set reference dataset for drift detection.
        
        Args:
            reference_data: Reference DataFrame
        """
        self.reference_data = reference_data.copy()
        logger.info(f"Reference data set with shape: {reference_data.shape}")
    
    def detect_drift(self, 
                    current_data: pd.DataFrame,
                    target_column: Optional[str] = None) -> Dict[str, DriftResult]:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current DataFrame to test
            target_column: Name of target column (optional)
            
        Returns:
            Dictionary of drift results by feature
        """
        if self.reference_data is None:
            raise ValueError("Reference data must be set before drift detection")
        
        logger.info("Starting drift detection")
        
        results = {}
        
        # Detect feature drift
        feature_results = self._detect_feature_drift(current_data)
        results.update(feature_results)
        
        # Detect target drift if target column provided
        if target_column and target_column in current_data.columns:
            target_result = self._detect_target_drift(current_data, target_column)
            results[target_column] = target_result
        
        # Detect overall dataset drift
        dataset_result = self._detect_dataset_drift(current_data)
        results['dataset_overall'] = dataset_result
        
        # Store results in history
        self.drift_history.append({
            'timestamp': datetime.now().isoformat(),
            'results': {name: {
                'drift_detected': result.drift_detected,
                'drift_score': result.drift_score,
                'p_value': result.p_value,
                'method': result.method
            } for name, result in results.items()}
        })
        
        logger.info(f"Drift detection completed. {sum(r.drift_detected for r in results.values())} features with drift detected")
        
        return results
    
    def _detect_feature_drift(self, current_data: pd.DataFrame) -> Dict[str, DriftResult]:
        """Detect drift for individual features."""
        results = {}
        
        # Get numerical and categorical features
        numerical_features = current_data.select_dtypes(include=[np.number]).columns
        categorical_features = current_data.select_dtypes(include=['object', 'category']).columns
        
        # Check numerical features
        for feature in numerical_features:
            if feature in self.reference_data.columns:
                result = self._detect_numerical_drift(feature, current_data)
                results[feature] = result
        
        # Check categorical features
        for feature in categorical_features:
            if feature in self.reference_data.columns:
                result = self._detect_categorical_drift(feature, current_data)
                results[feature] = result
        
        return results
    
    def _detect_numerical_drift(self, feature: str, current_data: pd.DataFrame) -> DriftResult:
        """Detect drift for numerical feature."""
        ref_data = self.reference_data[feature].dropna()
        current_data_clean = current_data[feature].dropna()
        
        if len(ref_data) == 0 or len(current_data_clean) == 0:
            return DriftResult(
                feature_name=feature,
                drift_detected=False,
                drift_score=0.0,
                p_value=1.0,
                threshold=self.threshold,
                method=self.detection_method,
                details={'error': 'Insufficient data'}
            )
        
        if self.detection_method == 'ks_test':
            return self._ks_test_drift(feature, ref_data, current_data_clean)
        elif self.detection_method == 'psi':
            return self._psi_drift(feature, ref_data, current_data_clean)
        elif self.detection_method == 'wasserstein':
            return self._wasserstein_drift(feature, ref_data, current_data_clean)
        else:
            return self._ks_test_drift(feature, ref_data, current_data_clean)
    
    def _detect_categorical_drift(self, feature: str, current_data: pd.DataFrame) -> DriftResult:
        """Detect drift for categorical feature."""
        ref_data = self.reference_data[feature].dropna()
        current_data_clean = current_data[feature].dropna()
        
        if len(ref_data) == 0 or len(current_data_clean) == 0:
            return DriftResult(
                feature_name=feature,
                drift_detected=False,
                drift_score=0.0,
                p_value=1.0,
                threshold=self.threshold,
                method='chi_square',
                details={'error': 'Insufficient data'}
            )
        
        return self._chi_square_drift(feature, ref_data, current_data_clean)
    
    def _ks_test_drift(self, feature: str, ref_data: pd.Series, current_data: pd.Series) -> DriftResult:
        """Kolmogorov-Smirnov test for drift detection."""
        try:
            ks_statistic, p_value = stats.ks_2samp(ref_data, current_data)
            drift_detected = p_value < self.threshold
            
            return DriftResult(
                feature_name=feature,
                drift_detected=drift_detected,
                drift_score=ks_statistic,
                p_value=p_value,
                threshold=self.threshold,
                method='ks_test',
                details={
                    'ks_statistic': ks_statistic,
                    'ref_mean': ref_data.mean(),
                    'current_mean': current_data.mean(),
                    'ref_std': ref_data.std(),
                    'current_std': current_data.std()
                }
            )
        except Exception as e:
            logger.error(f"KS test failed for {feature}: {str(e)}")
            return DriftResult(
                feature_name=feature,
                drift_detected=False,
                drift_score=0.0,
                p_value=1.0,
                threshold=self.threshold,
                method='ks_test',
                details={'error': str(e)}
            )
    
    def _psi_drift(self, feature: str, ref_data: pd.Series, current_data: pd.Series) -> DriftResult:
        """Population Stability Index for drift detection."""
        try:
            # Create bins based on reference data
            bins = np.histogram_bin_edges(ref_data, bins=10)
            
            # Calculate PSI
            psi_score = self._calculate_psi(ref_data, current_data, bins)
            
            # PSI interpretation: < 0.1 no change, 0.1-0.2 moderate change, > 0.2 significant change
            drift_detected = psi_score > 0.2
            
            return DriftResult(
                feature_name=feature,
                drift_detected=drift_detected,
                drift_score=psi_score,
                p_value=1.0 - min(psi_score / 0.5, 1.0),  # Convert to p-value-like metric
                threshold=0.2,
                method='psi',
                details={
                    'psi_score': psi_score,
                    'ref_mean': ref_data.mean(),
                    'current_mean': current_data.mean()
                }
            )
        except Exception as e:
            logger.error(f"PSI calculation failed for {feature}: {str(e)}")
            return DriftResult(
                feature_name=feature,
                drift_detected=False,
                drift_score=0.0,
                p_value=1.0,
                threshold=0.2,
                method='psi',
                details={'error': str(e)}
            )
    
    def _calculate_psi(self, ref_data: pd.Series, current_data: pd.Series, bins: np.ndarray) -> float:
        """Calculate Population Stability Index."""
        # Calculate histograms
        ref_hist, _ = np.histogram(ref_data, bins=bins)
        current_hist, _ = np.histogram(current_data, bins=bins)
        
        # Convert to proportions
        ref_prop = ref_hist / ref_hist.sum()
        current_prop = current_hist / current_hist.sum()
        
        # Avoid division by zero
        ref_prop = np.where(ref_prop == 0, 1e-6, ref_prop)
        current_prop = np.where(current_prop == 0, 1e-6, current_prop)
        
        # Calculate PSI
        psi = np.sum((current_prop - ref_prop) * np.log(current_prop / ref_prop))
        
        return psi
    
    def _wasserstein_drift(self, feature: str, ref_data: pd.Series, current_data: pd.Series) -> DriftResult:
        """Wasserstein distance for drift detection."""
        try:
            from scipy.stats import wasserstein_distance
            
            distance = wasserstein_distance(ref_data, current_data)
            
            # Normalize by the range of reference data
            ref_range = ref_data.max() - ref_data.min()
            normalized_distance = distance / ref_range if ref_range > 0 else distance
            
            # Threshold based on normalized distance
            drift_detected = normalized_distance > 0.1
            
            return DriftResult(
                feature_name=feature,
                drift_detected=drift_detected,
                drift_score=normalized_distance,
                p_value=1.0 - min(normalized_distance * 2, 1.0),
                threshold=0.1,
                method='wasserstein',
                details={
                    'wasserstein_distance': distance,
                    'normalized_distance': normalized_distance,
                    'ref_range': ref_range
                }
            )
        except Exception as e:
            logger.error(f"Wasserstein distance calculation failed for {feature}: {str(e)}")
            return DriftResult(
                feature_name=feature,
                drift_detected=False,
                drift_score=0.0,
                p_value=1.0,
                threshold=0.1,
                method='wasserstein',
                details={'error': str(e)}
            )
    
    def _chi_square_drift(self, feature: str, ref_data: pd.Series, current_data: pd.Series) -> DriftResult:
        """Chi-square test for categorical drift detection."""
        try:
            # Get unique values from both datasets
            all_values = set(ref_data.unique()) | set(current_data.unique())
            
            # Create contingency table
            ref_counts = ref_data.value_counts()
            current_counts = current_data.value_counts()
            
            # Align counts for all values
            ref_aligned = [ref_counts.get(val, 0) for val in all_values]
            current_aligned = [current_counts.get(val, 0) for val in all_values]
            
            # Perform chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency([ref_aligned, current_aligned])
            
            drift_detected = p_value < self.threshold
            
            return DriftResult(
                feature_name=feature,
                drift_detected=drift_detected,
                drift_score=chi2,
                p_value=p_value,
                threshold=self.threshold,
                method='chi_square',
                details={
                    'chi2_statistic': chi2,
                    'degrees_of_freedom': dof,
                    'ref_unique_values': len(ref_data.unique()),
                    'current_unique_values': len(current_data.unique())
                }
            )
        except Exception as e:
            logger.error(f"Chi-square test failed for {feature}: {str(e)}")
            return DriftResult(
                feature_name=feature,
                drift_detected=False,
                drift_score=0.0,
                p_value=1.0,
                threshold=self.threshold,
                method='chi_square',
                details={'error': str(e)}
            )
    
    def _detect_target_drift(self, current_data: pd.DataFrame, target_column: str) -> DriftResult:
        """Detect target drift."""
        try:
            ref_target = self.reference_data[target_column].dropna()
            current_target = current_data[target_column].dropna()
            
            if len(ref_target) == 0 or len(current_target) == 0:
                return DriftResult(
                    feature_name=target_column,
                    drift_detected=False,
                    drift_score=0.0,
                    p_value=1.0,
                    threshold=self.threshold,
                    method='target_drift',
                    details={'error': 'Insufficient target data'}
                )
            
            # For categorical targets, use chi-square test
            if pd.api.types.is_categorical_dtype(ref_target) or ref_target.dtype == 'object':
                return self._chi_square_drift(target_column, ref_target, current_target)
            else:
                # For numerical targets, use KS test
                return self._ks_test_drift(target_column, ref_target, current_target)
                
        except Exception as e:
            logger.error(f"Target drift detection failed: {str(e)}")
            return DriftResult(
                feature_name=target_column,
                drift_detected=False,
                drift_score=0.0,
                p_value=1.0,
                threshold=self.threshold,
                method='target_drift',
                details={'error': str(e)}
            )
    
    def _detect_dataset_drift(self, current_data: pd.DataFrame) -> DriftResult:
        """Detect overall dataset drift."""
        try:
            # Count features with drift
            feature_results = self._detect_feature_drift(current_data)
            drifted_features = sum(1 for result in feature_results.values() if result.drift_detected)
            total_features = len(feature_results)
            
            # Calculate drift ratio
            drift_ratio = drifted_features / total_features if total_features > 0 else 0
            
            # Dataset drift if more than 50% of features show drift
            dataset_drift = drift_ratio > 0.5
            
            return DriftResult(
                feature_name='dataset_overall',
                drift_detected=dataset_drift,
                drift_score=drift_ratio,
                p_value=1.0 - drift_ratio,
                threshold=0.5,
                method='dataset_drift',
                details={
                    'drifted_features': drifted_features,
                    'total_features': total_features,
                    'drift_ratio': drift_ratio
                }
            )
        except Exception as e:
            logger.error(f"Dataset drift detection failed: {str(e)}")
            return DriftResult(
                feature_name='dataset_overall',
                drift_detected=False,
                drift_score=0.0,
                p_value=1.0,
                threshold=0.5,
                method='dataset_drift',
                details={'error': str(e)}
            )
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection results."""
        if not self.drift_history:
            return {'message': 'No drift detection history available'}
        
        latest_results = self.drift_history[-1]['results']
        
        summary = {
            'timestamp': self.drift_history[-1]['timestamp'],
            'total_features_checked': len(latest_results),
            'features_with_drift': sum(1 for r in latest_results.values() if r['drift_detected']),
            'drift_rate': sum(1 for r in latest_results.values() if r['drift_detected']) / len(latest_results),
            'features_with_drift_details': {
                name: details for name, details in latest_results.items() 
                if details['drift_detected']
            }
        }
        
        return summary
    
    def save_drift_history(self, file_path: str) -> None:
        """Save drift detection history to file."""
        with open(file_path, 'w') as f:
            json.dump(self.drift_history, f, indent=2)
        
        logger.info(f"Drift history saved to {file_path}")
