"""
Production-grade data validator with comprehensive quality checks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Data class to store validation results"""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class DataValidator:
    """
    Comprehensive data validator for production ML pipelines.
    
    Features:
    - Schema validation
    - Data quality checks
    - Statistical validation
    - Business rule validation
    - Custom validation rules
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataValidator with configuration.
        
        Args:
            config: Configuration dictionary with validation rules
        """
        self.config = config
        self.data_quality_config = config.get('data_quality', {})
        self.validation_history = []
        
        logger.info("DataValidator initialized")
    
    def validate(self, df: pd.DataFrame, validation_type: str = 'full') -> ValidationResult:
        """
        Run comprehensive data validation.
        
        Args:
            df: DataFrame to validate
            validation_type: Type of validation ('full', 'quick', 'schema_only')
            
        Returns:
            ValidationResult object
        """
        result = ValidationResult(passed=True)
        
        try:
            logger.info(f"Starting {validation_type} validation")
            
            # Schema validation
            schema_result = self._validate_schema(df)
            result.errors.extend(schema_result.errors)
            result.warnings.extend(schema_result.warnings)
            
            if validation_type in ['full', 'quick']:
                # Missing value validation
                missing_result = self._validate_missing_values(df)
                result.errors.extend(missing_result.errors)
                result.warnings.extend(missing_result.warnings)
                
                # Data type validation
                dtype_result = self._validate_data_types(df)
                result.errors.extend(dtype_result.errors)
                result.warnings.extend(dtype_result.warnings)
                
                # Range validation
                range_result = self._validate_ranges(df)
                result.errors.extend(range_result.errors)
                result.warnings.extend(range_result.warnings)
            
            if validation_type == 'full':
                # Statistical validation
                stats_result = self._validate_statistics(df)
                result.warnings.extend(stats_result.warnings)
                
                # Outlier detection
                outlier_result = self._detect_outliers(df)
                result.warnings.extend(outlier_result.warnings)
                
                # Business rule validation
                business_result = self._validate_business_rules(df)
                result.errors.extend(business_result.errors)
            
            # Set overall status
            result.passed = len(result.errors) == 0
            
            # Calculate metrics
            result.metrics = self._calculate_validation_metrics(df, result)
            
            # Store validation history
            self.validation_history.append(result)
            
            logger.info(f"Validation completed. Passed: {result.passed}, "
                       f"Errors: {len(result.errors)}, Warnings: {len(result.warnings)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed with exception: {str(e)}")
            result.passed = False
            result.errors.append(f"Validation exception: {str(e)}")
            return result
    
    def _validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """Validate DataFrame schema against expected schema."""
        result = ValidationResult(passed=True)
        
        # Expected columns
        expected_columns = [
            'Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 
            'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT', 'Category'
        ]
        
        # Check for missing columns
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            result.errors.append(f"Missing required columns: {missing_cols}")
            result.passed = False
        
        # Check for extra columns
        extra_cols = set(df.columns) - set(expected_columns)
        if extra_cols:
            result.warnings.append(f"Unexpected columns found: {extra_cols}")
        
        return result
    
    def _validate_missing_values(self, df: pd.DataFrame) -> ValidationResult:
        """Validate missing values against configured thresholds."""
        result = ValidationResult(passed=True)
        
        threshold = self.data_quality_config.get('missing_value_threshold', 0.1)
        
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > threshold:
                result.errors.append(
                    f"Column '{col}' has {missing_pct:.2%} missing values "
                    f"(threshold: {threshold:.2%})"
                )
                result.passed = False
            elif missing_pct > 0:
                result.warnings.append(
                    f"Column '{col}' has {missing_pct:.2%} missing values"
                )
        
        return result
    
    def _validate_data_types(self, df: pd.DataFrame) -> ValidationResult:
        """Validate data types are appropriate."""
        result = ValidationResult(passed=True)
        
        # Expected data types
        numeric_cols = ['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
        categorical_cols = ['Sex', 'Category']
        
        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    result.errors.append(f"Column '{col}' should be numeric but is {df[col].dtype}")
                    result.passed = False
        
        for col in categorical_cols:
            if col in df.columns:
                if not pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
                    result.warnings.append(f"Column '{col}' should be categorical but is {df[col].dtype}")
        
        return result
    
    def _validate_ranges(self, df: pd.DataFrame) -> ValidationResult:
        """Validate that values are within expected ranges."""
        result = ValidationResult(passed=True)
        
        # Define valid ranges based on medical knowledge
        valid_ranges = {
            'Age': (0, 120),
            'ALB': (0, 10),  # g/dL
            'ALP': (0, 1000),  # U/L
            'ALT': (0, 1000),  # U/L
            'AST': (0, 1000),  # U/L
            'BIL': (0, 30),  # mg/dL
            'CHE': (0, 20),  # kU/L
            'CHOL': (0, 500),  # mg/dL
            'CREA': (0, 15),  # mg/dL
            'GGT': (0, 1500),  # U/L
            'PROT': (0, 15)  # g/dL
        }
        
        for col, (min_val, max_val) in valid_ranges.items():
            if col in df.columns:
                out_of_range = (df[col] < min_val) | (df[col] > max_val)
                count = out_of_range.sum()
                if count > 0:
                    result.warnings.append(
                        f"Column '{col}' has {count} values outside valid range [{min_val}, {max_val}]"
                    )
        
        # Validate categorical values
        if 'Sex' in df.columns:
            valid_sex = ['m', 'f', 'M', 'F']
            invalid = ~df['Sex'].isin(valid_sex)
            if invalid.sum() > 0:
                result.errors.append(f"Invalid 'Sex' values found: {df.loc[invalid, 'Sex'].unique()}")
                result.passed = False
        
        if 'Category' in df.columns:
            valid_categories = ['Blood donor', 'Cirrhosis', 'Fibrosis', 'Hepatitis']
            invalid = ~df['Category'].isin(valid_categories)
            if invalid.sum() > 0:
                result.errors.append(f"Invalid 'Category' values found: {df.loc[invalid, 'Category'].unique()}")
                result.passed = False
        
        return result
    
    def _validate_statistics(self, df: pd.DataFrame) -> ValidationResult:
        """Validate statistical properties of the data."""
        result = ValidationResult(passed=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Check for constant columns
            if df[col].nunique() == 1:
                result.warnings.append(f"Column '{col}' has constant value")
            
            # Check for extreme skewness
            try:
                skewness = df[col].skew()
                if abs(skewness) > 5:
                    result.warnings.append(f"Column '{col}' has high skewness: {skewness:.2f}")
            except:
                pass
        
        return result
    
    def _detect_outliers(self, df: pd.DataFrame) -> ValidationResult:
        """Detect outliers using statistical methods."""
        result = ValidationResult(passed=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_pct = outliers / len(df) * 100
            
            if outlier_pct > 10:
                result.warnings.append(
                    f"Column '{col}' has {outlier_pct:.2f}% potential outliers"
                )
        
        return result
    
    def _validate_business_rules(self, df: pd.DataFrame) -> ValidationResult:
        """Validate business-specific rules."""
        result = ValidationResult(passed=True)
        
        # Example: Age should be positive
        if 'Age' in df.columns:
            if (df['Age'] <= 0).any():
                result.errors.append("Age must be positive")
                result.passed = False
        
        # Example: ALT and AST relationship (usually ALT <= AST in liver disease)
        if 'ALT' in df.columns and 'AST' in df.columns:
            ratio = (df['ALT'] > df['AST'] * 2).sum() / len(df)
            if ratio > 0.5:
                result.warnings.append(f"Unusual ALT/AST ratio in {ratio:.2%} of cases")
        
        return result
    
    def _calculate_validation_metrics(self, df: pd.DataFrame, result: ValidationResult) -> Dict[str, Any]:
        """Calculate validation metrics."""
        metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values_total': df.isnull().sum().sum(),
            'missing_values_pct': (df.isnull().sum().sum() / df.size * 100),
            'duplicate_rows': df.duplicated().sum(),
            'validation_passed': result.passed,
            'error_count': len(result.errors),
            'warning_count': len(result.warnings)
        }
        
        return metrics
    
    def get_validation_report(self, result: ValidationResult) -> str:
        """
        Generate a human-readable validation report.
        
        Args:
            result: ValidationResult object
            
        Returns:
            Formatted validation report string
        """
        report = []
        report.append("=" * 80)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {result.timestamp}")
        report.append(f"Status: {'PASSED' if result.passed else 'FAILED'}")
        report.append(f"\nMetrics:")
        for key, value in result.metrics.items():
            report.append(f"  {key}: {value}")
        
        if result.errors:
            report.append(f"\nErrors ({len(result.errors)}):")
            for error in result.errors:
                report.append(f"  ❌ {error}")
        
        if result.warnings:
            report.append(f"\nWarnings ({len(result.warnings)}):")
            for warning in result.warnings:
                report.append(f"  ⚠️  {warning}")
        
        report.append("=" * 80)
        
        return "\n".join(report)
