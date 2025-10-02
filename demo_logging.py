#!/usr/bin/env python3
"""
Demo script to showcase the improved logging functionality
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path so we can import our module
sys.path.append(os.getcwd())

from liver_disease_prediction_python import LiverDiseasePredictor, setup_logging
import logging

def demo_logging():
    """
    Demonstrate the enhanced logging capabilities
    """
    print("=" * 80)
    print("DEMONSTRATING ENHANCED LOGGING CAPABILITIES")
    print("=" * 80)
    
    # Setup logging with different levels
    print("\n1. Setting up comprehensive logging...")
    logger = setup_logging(log_level="DEBUG")
    
    # Demonstrate different log levels
    print("\n2. Demonstrating different log levels...")
    logger.debug("This is a DEBUG message - detailed information for debugging")
    logger.info("This is an INFO message - general information about the process")
    logger.warning("This is a WARNING message - something unexpected happened")
    logger.error("This is an ERROR message - an error occurred")
    logger.critical("This is a CRITICAL message - a serious error occurred")
    
    # Demonstrate structured logging
    print("\n3. Demonstrating structured logging...")
    structured_logger = logging.getLogger("structured")
    
    # Log some sample metrics
    sample_metrics = {
        "model_name": "Random Forest",
        "accuracy": 0.95,
        "precision": 0.93,
        "recall": 0.94,
        "f1_score": 0.935
    }
    
    structured_logger.info(f'{{"event": "model_evaluation", "metrics": {sample_metrics}, "timestamp": "{__import__("datetime").datetime.now().isoformat()}"}}')
    
    # Demonstrate performance logging
    print("\n4. Demonstrating performance tracking...")
    
    # Create a small predictor instance to show logging in action
    print("\n5. Running a quick demo of the Liver Disease Predictor with logging...")
    
    try:
        predictor = LiverDiseasePredictor("sample_hcv_data.csv")  # Use small sample
        
        # Just load data to show logging in action
        predictor.load_data()
        
        print("\n6. Logging demonstration completed!")
        print("Check the 'output' directory for:")
        print("- Detailed log file with timestamps")
        print("- Structured JSON log file for metrics")
        print("- Performance tracking information")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
    
    print("\n" + "=" * 80)
    print("LOGGING DEMO COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    demo_logging()
