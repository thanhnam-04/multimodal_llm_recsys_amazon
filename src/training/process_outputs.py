#!/usr/bin/env python3
"""
Process model outputs for evaluation.
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.process_model_outputs import main as process_outputs_main

def main():
    """Process model outputs."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Processing model outputs...")
    
    try:
        # Run the process outputs
        process_outputs_main()
        
        logger.info("Model output processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error processing model outputs: {e}")
        raise

if __name__ == "__main__":
    main()
