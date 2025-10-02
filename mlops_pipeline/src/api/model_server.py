"""
Production-grade FastAPI model server with comprehensive features
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import joblib
from pathlib import Path
import uvicorn

from .schemas import PredictionRequest, PredictionResponse, HealthResponse, ModelInfo
from .middleware import RateLimitMiddleware, LoggingMiddleware
from .health_check import HealthChecker

logger = logging.getLogger(__name__)

class ModelServer:
    """
    Production-grade model serving API with comprehensive features.
    
    Features:
    - Model loading and caching
    - Request/response validation
    - Rate limiting
    - Health checks
    - Metrics collection
    - Error handling
    - Logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelServer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.api_config = config.get('api', {})
        self.models = {}
        self.model_metadata = {}
        self.request_count = 0
        self.start_time = datetime.now()
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Liver Disease Prediction API",
            description="Production ML model serving API for liver disease prediction",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add middleware
        self._setup_middleware()
        
        # Add routes
        self._setup_routes()
        
        # Initialize health checker
        self.health_checker = HealthChecker(config)
        
        logger.info("ModelServer initialized")
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]
        )
        
        # Custom middleware
        self.app.add_middleware(LoggingMiddleware)
        self.app.add_middleware(RateLimitMiddleware, rate_limit=self.api_config.get('rate_limit', 100))
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint."""
            return {
                "message": "Liver Disease Prediction API",
                "version": "1.0.0",
                "docs": "/docs"
            }
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return await self._health_check()
        
        @self.app.get("/models", response_model=Dict[str, ModelInfo])
        async def list_models():
            """List available models."""
            return self._list_models()
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Make predictions."""
            return await self._predict(request)
        
        @self.app.post("/predict_batch", response_model=List[PredictionResponse])
        async def predict_batch(requests: List[PredictionRequest]):
            """Make batch predictions."""
            return await self._predict_batch(requests)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get API metrics."""
            return self._get_metrics()
        
        @self.app.post("/models/{model_name}/reload")
        async def reload_model(model_name: str):
            """Reload a specific model."""
            return await self._reload_model(model_name)
    
    def load_model(self, model_name: str, model_path: str) -> None:
        """
        Load a model into memory.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model file
        """
        try:
            logger.info(f"Loading model: {model_name} from {model_path}")
            
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model
            model_data = joblib.load(model_path)
            
            if isinstance(model_data, dict):
                model = model_data.get('model')
                metadata = model_data.get('metadata', {})
            else:
                model = model_data
                metadata = {}
            
            # Store model and metadata
            self.models[model_name] = model
            self.model_metadata[model_name] = {
                'path': str(model_path),
                'loaded_at': datetime.now().isoformat(),
                'version': metadata.get('version', '1.0.0'),
                'performance': metadata.get('performance', {}),
                **metadata
            }
            
            logger.info(f"Model {model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    async def _health_check(self) -> HealthResponse:
        """Perform health check."""
        try:
            health_status = await self.health_checker.check_health()
            
            return HealthResponse(
                status="healthy" if health_status['overall'] else "unhealthy",
                timestamp=datetime.now().isoformat(),
                uptime=time.time() - self.start_time.timestamp(),
                models_loaded=len(self.models),
                details=health_status
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return HealthResponse(
                status="unhealthy",
                timestamp=datetime.now().isoformat(),
                uptime=time.time() - self.start_time.timestamp(),
                models_loaded=len(self.models),
                error=str(e)
            )
    
    def _list_models(self) -> Dict[str, ModelInfo]:
        """List available models."""
        models = {}
        for name, metadata in self.model_metadata.items():
            models[name] = ModelInfo(
                name=name,
                version=metadata.get('version', '1.0.0'),
                loaded_at=metadata.get('loaded_at'),
                performance=metadata.get('performance', {}),
                status="loaded" if name in self.models else "unloaded"
            )
        
        return models
    
    async def _predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make single prediction."""
        try:
            start_time = time.time()
            
            # Validate request
            if not request.model_name or request.model_name not in self.models:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{request.model_name}' not found"
                )
            
            # Get model
            model = self.models[request.model_name]
            
            # Prepare input data
            input_data = self._prepare_input_data(request.features)
            
            # Make prediction
            prediction = model.predict(input_data)
            prediction_proba = None
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(input_data).tolist()
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update request count
            self.request_count += 1
            
            return PredictionResponse(
                prediction=prediction.tolist() if hasattr(prediction, 'tolist') else [prediction],
                prediction_proba=prediction_proba,
                model_name=request.model_name,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _predict_batch(self, requests: List[PredictionRequest]) -> List[PredictionResponse]:
        """Make batch predictions."""
        try:
            start_time = time.time()
            
            # Group requests by model
            model_requests = {}
            for i, request in enumerate(requests):
                if request.model_name not in model_requests:
                    model_requests[request.model_name] = []
                model_requests[request.model_name].append((i, request))
            
            # Process each model
            responses = [None] * len(requests)
            
            for model_name, model_requests_list in model_requests.items():
                if model_name not in self.models:
                    # Set error responses for this model
                    for idx, _ in model_requests_list:
                        responses[idx] = PredictionResponse(
                            prediction=None,
                            prediction_proba=None,
                            model_name=model_name,
                            processing_time=0,
                            timestamp=datetime.now().isoformat(),
                            error=f"Model '{model_name}' not found"
                        )
                    continue
                
                model = self.models[model_name]
                
                # Prepare batch input
                batch_features = []
                indices = []
                for idx, request in model_requests_list:
                    batch_features.append(request.features)
                    indices.append(idx)
                
                # Make batch prediction
                batch_input = pd.DataFrame(batch_features)
                batch_predictions = model.predict(batch_input)
                batch_proba = None
                
                if hasattr(model, 'predict_proba'):
                    batch_proba = model.predict_proba(batch_input)
                
                # Create responses
                for i, idx in enumerate(indices):
                    responses[idx] = PredictionResponse(
                        prediction=batch_predictions[i].tolist() if hasattr(batch_predictions[i], 'tolist') else [batch_predictions[i]],
                        prediction_proba=batch_proba[i].tolist() if batch_proba is not None else None,
                        model_name=model_name,
                        processing_time=(time.time() - start_time) / len(requests),
                        timestamp=datetime.now().isoformat()
                    )
            
            return responses
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _prepare_input_data(self, features: Dict[str, Any]) -> pd.DataFrame:
        """Prepare input data for model prediction."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([features])
            
            # Validate required features
            required_features = ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 
                               'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
            
            missing_features = set(required_features) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to prepare input data: {str(e)}")
            raise
    
    def _get_metrics(self) -> Dict[str, Any]:
        """Get API metrics."""
        uptime = time.time() - self.start_time.timestamp()
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "requests_per_minute": self.request_count / (uptime / 60) if uptime > 0 else 0,
            "models_loaded": len(self.models),
            "memory_usage": self._get_memory_usage(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    async def _reload_model(self, model_name: str) -> Dict[str, str]:
        """Reload a specific model."""
        try:
            if model_name not in self.model_metadata:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
            
            metadata = self.model_metadata[model_name]
            model_path = metadata['path']
            
            # Reload model
            self.load_model(model_name, model_path)
            
            return {"message": f"Model '{model_name}' reloaded successfully"}
            
        except Exception as e:
            logger.error(f"Failed to reload model {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, host: str = None, port: int = None, workers: int = None):
        """
        Run the model server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            workers: Number of worker processes
        """
        host = host or self.api_config.get('host', '0.0.0.0')
        port = port or self.api_config.get('port', 8000)
        workers = workers or self.api_config.get('workers', 1)
        
        logger.info(f"Starting model server on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=workers,
            log_level="info"
        )
