#!/usr/bin/env python3
"""
Enhanced Multimodal LLM Recommendation Workflow
Optimized for Amazon product recommendations with multimodal (text + image) processing
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


def find_project_root() -> Path:
    """Find the project root directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "configs" / "train_config.json").exists():
            return current
        current = current.parent
    return Path.cwd()


def setup_environment():
    """Setup the environment and change to project directory."""
    project_root = find_project_root()
    os.chdir(project_root)
    return project_root


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return {}


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    logger.info("=" * 60)
    
    try:
        # Always run with the currently active interpreter.
        python_path = sys.executable
        if command == "python" or command.startswith("python "):
            command = command.replace("python", python_path, 1)
        
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.getcwd(),
            bufsize=1
        )

        if process.stdout is not None:
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    logger.info(line)

        return_code = process.wait()

        if return_code == 0:
            logger.info("Command executed successfully!")
            return True
        else:
            logger.error("Command failed!")
            logger.error(f"Command exited with code {return_code}")
            return False
            
    except Exception as e:
        logger.error(f"Exception running command: {e}")
        return False


def main():
    """Main workflow function."""
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    
    # Setup logging to use train.log only
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/train.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Enhanced Multimodal LLM Recommendation Workflow")
    logger.info("=" * 60)
    logger.info(" Optimized multimodal processing enabled:")
    logger.info("   - Batch image downloading (20 concurrent)")
    logger.info("   - Fast image validation (no HTTP requests)")
    logger.info("   - Connection pooling and shorter timeouts")
    logger.info("   - Image caching for faster subsequent runs")
    logger.info("=" * 60)
    
    # Setup environment
    project_root = setup_environment()
    logger.info(f"Project root: {project_root}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Enhanced Multimodal LLM Recommendation Workflow")
    parser.add_argument("--skip-training", action="store_true", help="Skip training step")
    parser.add_argument("--steps", nargs="+", help="Run specific steps only", 
                       choices=["data", "train", "process", "eval", "baseline", "cross", "all"])
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--force-training", action="store_true", help="Force retraining even if model exists")
    parser.add_argument("--config", help="Path to config file", default="configs/train_config.json")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config if args.config else "configs/train_config.json"
    config = load_config(config_path)
    
    if not config:
        logger.error("Failed to load configuration")
        return
    
    logger.info(f"Successfully loaded configuration from: {config_path}")
    logger.info("Configuration:")
    logger.info(f"  - Multi-category mode: {config.get('data_config', {}).get('use_multiple_categories', False)}")
    logger.info(f"  - Statistical testing: {config.get('training_config', {}).get('statistical_testing', False)}")
    logger.info(f"  - Evaluation runs: {config.get('training_config', {}).get('evaluation_runs', 1)}")
    logger.info(f"  - Ablation studies: {config.get('training_config', {}).get('ablation_studies', False)}")
    logger.info(f"  - Dataset size: {config.get('model_config', {}).get('test_size', 'N/A')}")
    
    # Determine which steps to run
    if args.steps:
        steps_to_run = args.steps
        logger.info(f"Running specific steps: {' '.join(steps_to_run)}")
    else:
        steps_to_run = ["data", "train", "process", "eval", "baseline", "cross"]
    
    # Step 1: Data Preparation
    if "data" in steps_to_run:
        logger.info("Step 1: Data Preparation")
        logger.info("=" * 60)
        success = run_command(
            "python -m src.data.prepare_data",
            "Data preparation with multi-category support"
        )
        if not success:
            logger.error("Data preparation failed. Exiting.")
            return
    
    # Step 2: Model Training
    if "train" in steps_to_run and not args.skip_training:
        logger.info("Step 2: Model Training")
        logger.info("=" * 60)
        success = run_command(
            "python -m src.training.train",
            "Training the multimodal LLM model"
        )
        if not success:
            logger.error("Training failed. Exiting.")
            return
    else:
        logger.info("Step 2: Model Training (skipped)")
    
    # Step 3: Model Output Processing
    if "process" in steps_to_run:
        logger.info("Step 3: Model Output Processing")
        logger.info("=" * 60)
        success = run_command(
            "python -m src.training.process_outputs",
            "Processing model outputs for evaluation"
        )
        if not success:
            logger.error("Output processing failed. Exiting.")
            return
    else:
        logger.info("Step 3: Model Output Processing (skipped)")
    
    # Step 4: Baseline Comparison
    if "baseline" in steps_to_run:
        logger.info("Step 4: Baseline Comparison")
        logger.info("=" * 60)
        success = run_command(
            "python -m src.baselines.run_baselines",
            "Running baseline models for comparison"
        )
        if not success:
            logger.error("Baseline comparison failed. Exiting.")
            return
    else:
        logger.info("Step 4: Baseline Comparison (skipped)")
    
    # Step 5: Basic Evaluation
    if "eval" in steps_to_run:
        processed_predictions = Path("data/processed/test_with_responses_processed.json")
        if not processed_predictions.exists():
            logger.info("Processed prediction file missing; running output processing first...")
            success = run_command(
                "python -m src.training.process_outputs",
                "Processing model outputs required by evaluation"
            )
            if not success:
                logger.error("Output processing failed before evaluation. Exiting.")
                return

        logger.info("Step 5: Basic Evaluation")
        logger.info("=" * 60)
        success = run_command(
            "python -m src.evaluation.basic_evaluation",
            "Running basic evaluation with standard metrics"
        )
        if not success:
            logger.error("Basic evaluation failed. Exiting.")
            return
    else:
        logger.info("Step 5: Basic Evaluation (skipped)")
    
    # Step 6: Ablation Studies
    if "eval" in steps_to_run:
        logger.info("Step 6: Ablation Studies")
        logger.info("=" * 60)
        success = run_command(
            "python -m src.evaluation.ablation_studies",
            "Running ablation studies"
        )
        if not success:
            logger.error("Ablation studies failed. Exiting.")
            return
    else:
        logger.info("Step 7: Ablation Studies (skipped)")
    
    # Step 7: Cross-Category Analysis
    if "cross" in steps_to_run:
        logger.info("Step 7: Cross-Category Analysis")
        logger.info("=" * 60)
        success = run_command(
            "python -m src.evaluation.cross_category_evaluation",
            "Running cross-category evaluation"
        )
        if not success:
            logger.error("Cross-category analysis failed. Exiting.")
            return
    else:
        logger.info("Step 7: Cross-Category Analysis (skipped)")
    
    # Step 8: Main Evaluation (Generate Comprehensive Results)
    if "eval" in steps_to_run:
        logger.info("Step 8: Main Evaluation")
        logger.info("=" * 60)
        success = run_command(
            "python -m src.evaluation.main_evaluation",
            "Generating comprehensive evaluation results"
        )
        if not success:
            logger.error("Main evaluation failed. Exiting.")
            return
    else:
        logger.info("Step 8: Main Evaluation (skipped)")
    
    # Step 9: Generate Summary Report
    logger.info("Step 9: Generating Summary Report")
    logger.info("=" * 60)
    
    # Check for results files
    results_dir = Path("results")
    result_files = {
        "Basic Evaluation": results_dir / "basic_evaluation_results.json",
        "Main Results": results_dir / "result_metrics.json",
        "Ablation Studies": results_dir / "ablation_study_results.json",
        "Baseline Predictions": results_dir / "test_with_baseline_predictions.json",
        "Cross Category Analysis": results_dir / "cross_category_evaluation_results.json",
        "Cross Category Summary": results_dir / "cross_category_summary.json"
    }
    
    for name, file_path in result_files.items():
        if file_path.exists():
            logger.info(f"  {name}: {file_path}")
        else:
            logger.warning(f"  {name} not found")
    
    logger.info("Enhanced workflow completed successfully!")
    logger.info("Results saved to:")
    logger.info("  - Basic evaluation: data/processed/basic_evaluation_results.json")
    logger.info("  - Main results: data/processed/result_metrics.json")
    logger.info("  - Ablation studies: data/processed/ablation_study_results.json")
    logger.info("  - Baseline predictions: data/processed/test_with_baseline_predictions.json")
    logger.info("  - Cross-category analysis: data/processed/cross_category_evaluation_results.json")
    logger.info("  - Cross-category summary: data/processed/cross_category_summary.json")
    logger.info("  - Performance visualizations: output/plots/")


def run_from_anywhere():
    """Function to run the workflow from any directory."""
    main()


if __name__ == "__main__":
    run_from_anywhere()
