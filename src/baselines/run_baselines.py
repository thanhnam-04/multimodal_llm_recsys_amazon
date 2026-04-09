#!/usr/bin/env python3
"""
Run baseline models for comparison with the multimodal LLM.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.baselines.multimodal_baselines import run_all_baselines
from src.utils.utils import load_config

def main():
    """Run all baseline models."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Running baseline models...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Run all baselines
        results = run_all_baselines(config)
        
        logger.info("Baseline models completed successfully!")
        logger.info(f"Results saved to: data/processed/test_with_baseline_predictions.json")
        
    except Exception as e:
        logger.error(f"Error running baseline models: {e}")
        raise

if __name__ == "__main__":
    main()
