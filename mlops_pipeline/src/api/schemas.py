"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

class PredictionRequest(BaseModel):
    """Schema for prediction requests."""
    
    model_name: str = Field(..., description="Name of the model to use")
    features: Dict[str, Union[float, int, str]] = Field(..., description="Input features")
    
    @validator('features')
    def validate_features(cls, v):
        """Validate that required features are present."""
        required_features = {
            'Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 
            'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT'
        }
        
        if not isinstance(v, dict):
            raise ValueError("Features must be a dictionary")
        
        missing = required_features - set(v.keys())
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        return v
    
    @validator('Age')
    def validate_age(cls, v):
        """Validate age is positive."""
        if v <= 0 or v > 120:
            raise ValueError("Age must be between 1 and 120")
        return v
    
    @validator('Sex')
    def validate_sex(cls, v):
        """Validate sex values."""
        if v not in ['m', 'f', 'M', 'F']:
            raise ValueError("Sex must be 'm' or 'f'")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "liver_disease_classifier",
                "features": {
                    "Age": 45,
                    "Sex": "m",
                    "ALB": 4.0,
                    "ALP": 100,
                    "ALT": 40,
                    "AST": 35,
                    "BIL": 1.0,
                    "CHE": 8.0,
                    "CHOL": 200,
                    "CREA": 1.1,
                    "GGT": 45,
                    "PROT": 7.2
                }
            }
        }

class PredictionResponse(BaseModel):
    """Schema for prediction responses."""
    
    prediction: Optional[List[Union[str, int]]] = Field(None, description="Model predictions")
    prediction_proba: Optional[List[List[float]]] = Field(None, description="Prediction probabilities")
    model_name: str = Field(..., description="Name of the model used")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Timestamp of prediction")
    error: Optional[str] = Field(None, description="Error message if prediction failed")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": ["Blood donor"],
                "prediction_proba": [[0.8, 0.1, 0.05, 0.05]],
                "model_name": "liver_disease_classifier",
                "processing_time": 0.023,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }

class HealthResponse(BaseModel):
    """Schema for health check responses."""
    
    status: str = Field(..., description="Overall health status")
    timestamp: str = Field(..., description="Timestamp of health check")
    uptime: float = Field(..., description="Server uptime in seconds")
    models_loaded: int = Field(..., description="Number of models loaded")
    details: Optional[Dict[str, Any]] = Field(None, description="Detailed health information")
    error: Optional[str] = Field(None, description="Error message if health check failed")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z",
                "uptime": 3600.0,
                "models_loaded": 2,
                "details": {
                    "database": "healthy",
                    "models": "healthy",
                    "memory": "healthy"
                }
            }
        }

class ModelInfo(BaseModel):
    """Schema for model information."""
    
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    loaded_at: Optional[str] = Field(None, description="When model was loaded")
    performance: Dict[str, float] = Field(default_factory=dict, description="Model performance metrics")
    status: str = Field(..., description="Model status")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "liver_disease_classifier",
                "version": "1.0.0",
                "loaded_at": "2024-01-01T12:00:00Z",
                "performance": {
                    "accuracy": 0.95,
                    "precision": 0.93,
                    "recall": 0.94,
                    "f1_score": 0.935
                },
                "status": "loaded"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests."""
    
    model_name: str = Field(..., description="Name of the model to use")
    features_list: List[Dict[str, Union[float, int, str]]] = Field(..., description="List of input features")
    
    @validator('features_list')
    def validate_features_list(cls, v):
        """Validate features list."""
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("features_list must be a non-empty list")
        
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000")
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "liver_disease_classifier",
                "features_list": [
                    {
                        "Age": 45,
                        "Sex": "m",
                        "ALB": 4.0,
                        "ALP": 100,
                        "ALT": 40,
                        "AST": 35,
                        "BIL": 1.0,
                        "CHE": 8.0,
                        "CHOL": 200,
                        "CREA": 1.1,
                        "GGT": 45,
                        "PROT": 7.2
                    },
                    {
                        "Age": 50,
                        "Sex": "f",
                        "ALB": 3.8,
                        "ALP": 120,
                        "ALT": 50,
                        "AST": 40,
                        "BIL": 1.2,
                        "CHE": 7.5,
                        "CHOL": 180,
                        "CREA": 1.0,
                        "GGT": 55,
                        "PROT": 7.0
                    }
                ]
            }
        }

class ModelMetrics(BaseModel):
    """Schema for model performance metrics."""
    
    model_name: str = Field(..., description="Model name")
    accuracy: float = Field(..., description="Accuracy score")
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    f1_score: float = Field(..., description="F1 score")
    auc_roc: Optional[float] = Field(None, description="AUC-ROC score")
    confusion_matrix: Optional[List[List[int]]] = Field(None, description="Confusion matrix")
    timestamp: str = Field(..., description="Timestamp when metrics were calculated")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "liver_disease_classifier",
                "accuracy": 0.95,
                "precision": 0.93,
                "recall": 0.94,
                "f1_score": 0.935,
                "auc_roc": 0.96,
                "confusion_matrix": [[50, 2], [1, 47]],
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }

class DataDriftReport(BaseModel):
    """Schema for data drift detection report."""
    
    feature_name: str = Field(..., description="Name of the feature")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    drift_score: float = Field(..., description="Drift score")
    p_value: float = Field(..., description="Statistical p-value")
    threshold: float = Field(..., description="Drift detection threshold")
    timestamp: str = Field(..., description="Timestamp of drift check")
    
    class Config:
        schema_extra = {
            "example": {
                "feature_name": "Age",
                "drift_detected": False,
                "drift_score": 0.12,
                "p_value": 0.45,
                "threshold": 0.05,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }

class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Timestamp when error occurred")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Model not found",
                "detail": "The requested model 'invalid_model' is not available",
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req_123456"
            }
        }
