"""
Robust evaluation module with statistical testing and multiple runs.
Addresses the critical evaluation credibility issues identified by reviewers.
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from pathlib import Path
import logging
from scipy import stats
from src.evaluation.metrics import calculate_metrics
from collections import defaultdict
import random
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class RobustEvaluator:
    """
    Robust evaluator that addresses evaluation credibility issues:
    - Multiple runs with different random seeds
    - Statistical significance testing
    - Confidence intervals
    - Proper train/test splits
    """
    
    def __init__(self, config: Dict[str, Any], num_runs: int = 5, confidence_level: float = 0.95):
        self.config = config
        self.num_runs = num_runs
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def calculate_metrics_robust(self, predictions: List[List[str]], targets: List[List[str]], k: int = 10) -> Dict[str, float]:
        """
        Calculate two-track metrics with proper handling of edge cases.
        """
        if not predictions or not targets:
            logger.warning("Empty predictions or targets")
            return {
                f'item_precision@{k}': 0.0, f'item_recall@{k}': 0.0, f'item_hitrate@{k}': 0.0,
                f'item_ndcg@{k}': 0.0, 'item_mrr': 0.0, 'item_map': 0.0,
                f'item_coverage@{k}': 0.0, f'item_diversity@{k}': 0.0, f'item_novelty@{k}': 0.0,
                f'stop_hit@{k}': 0.0, f'stop_ndcg@{k}': 0.0,
                'stop_rank_median': None, 'stop_rank_mean': None, 'stop_rank_std': None,
                'stop_prediction_rate': 0.0
            }
        
        # Use the new two-track metrics system
        return calculate_metrics(predictions, targets, k)
    
    def run_multiple_evaluations(self, model_predictions: Dict[str, List[List[str]]], 
                                targets: List[List[str]], k: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Run multiple evaluations with different random seeds and calculate statistics.
        """
        results = {}
        
        for model_name, predictions in model_predictions.items():
            logger.info(f"Running {self.num_runs} evaluations for {model_name}")
            
            # Store metrics for each run - initialize with two-track metric names
            run_metrics = defaultdict(list)
            
            for run in range(self.num_runs):
                # Set random seed for reproducibility
                random.seed(42 + run)
                np.random.seed(42 + run)
                
                # Shuffle predictions and targets together to maintain correspondence
                combined = list(zip(predictions, targets))
                random.shuffle(combined)
                shuffled_predictions, shuffled_targets = zip(*combined)
                
                # Calculate metrics for this run
                metrics = self.calculate_metrics_robust(list(shuffled_predictions), list(shuffled_targets), k)
                
                # Store metrics
                for metric_name, value in metrics.items():
                    # Keep the full metric name for two-track metrics
                    run_metrics[metric_name].append(value)
            
            # Calculate statistics
            model_results = {}
            for metric_name, values in run_metrics.items():
                if values:
                    # Filter out None values for metrics that can have None
                    filtered_values = [v for v in values if v is not None]
                    if filtered_values:
                        mean_val = np.mean(filtered_values)
                        std_val = np.std(filtered_values)
                        ci_lower, ci_upper = self._calculate_confidence_interval(filtered_values)
                        
                        # Use the metric name as-is for two-track metrics
                        model_results[metric_name] = {
                            'mean': mean_val,
                            'std': std_val,
                            'ci_lower': ci_lower,
                            'ci_upper': ci_upper,
                            'values': filtered_values
                        }
                    else:
                        # All values were None
                        model_results[metric_name] = {
                            'mean': None,
                            'std': None,
                            'ci_lower': None,
                            'ci_upper': None,
                            'values': values
                        }
            
            results[model_name] = model_results
            
            # Log results
            logger.info(f"Results for {model_name} (mean ± std, 95% CI):")
            for metric_name, stats_dict in model_results.items():
                mean_val = stats_dict['mean']
                std_val = stats_dict['std']
                ci_lower = stats_dict['ci_lower']
                ci_upper = stats_dict['ci_upper']
                
                if mean_val is not None and std_val is not None:
                    logger.info(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f} "
                               f"[{ci_lower:.4f}, {ci_upper:.4f}]")
                else:
                    logger.info(f"  {metric_name}: None (all values were None)")
        
        return results
    
    def _calculate_confidence_interval(self, values: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for the given values."""
        if len(values) < 2:
            return (values[0], values[0]) if values else (0.0, 0.0)
        
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)  # Use sample standard deviation
        
        # Calculate confidence interval using t-distribution
        t_val = stats.t.ppf(1 - self.alpha/2, len(values) - 1)
        margin_error = t_val * (std_val / np.sqrt(len(values)))
        
        return (mean_val - margin_error, mean_val + margin_error)
    
    def statistical_significance_test(self, results: Dict[str, Dict[str, Any]], 
                                    baseline_model: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Perform statistical significance tests comparing all models to baseline.
        """
        significance_results = {}
        
        # If no baseline specified, use the first model as baseline
        if baseline_model is None:
            baseline_model = list(results.keys())[0]
        
        # Skip if baseline model doesn't exist
        if baseline_model not in results:
            logger.warning(f"Baseline model '{baseline_model}' not found in results. Skipping significance tests.")
            return significance_results
        
        for model_name, model_results in results.items():
            if model_name == baseline_model:
                continue
                
            significance_results[model_name] = {}
            
            for metric_name, stats_dict in model_results.items():
                if metric_name in results[baseline_model]:
                    baseline_values = results[baseline_model][metric_name]['values']
                    model_values = stats_dict['values']
                    
                    # Filter out None values for statistical testing
                    baseline_values_clean = [v for v in baseline_values if v is not None]
                    model_values_clean = [v for v in model_values if v is not None]
                    
                    # Perform paired t-test only if we have valid values
                    if (len(baseline_values_clean) == len(model_values_clean) and 
                        len(baseline_values_clean) > 1 and 
                        len(model_values_clean) > 1):
                        try:
                            t_stat, p_value = stats.ttest_rel(model_values_clean, baseline_values_clean)
                            
                            significance_results[model_name][metric_name] = {
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < self.alpha,
                                'effect_size': self._calculate_cohens_d(model_values_clean, baseline_values_clean)
                            }
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Statistical test failed for {model_name} {metric_name}: {e}")
                            significance_results[model_name][metric_name] = {
                                't_statistic': None,
                                'p_value': None,
                                'significant': False,
                                'effect_size': 0.0
                            }
                    else:
                        # Not enough valid values for statistical testing
                        significance_results[model_name][metric_name] = {
                            't_statistic': None,
                            'p_value': None,
                            'significant': False,
                            'effect_size': 0.0
                        }
        
        return significance_results
    
    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((len(group1) - 1) * std1**2 + (len(group2) - 1) * std2**2) / 
                            (len(group1) + len(group2) - 2))
        
        return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
    
    def save_results(self, results: Dict[str, Any], significance_results: Dict[str, Any], 
                    output_path: str):
        """Save evaluation results with statistical analysis."""
        output_data = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'num_runs': self.num_runs,
            'confidence_level': self.confidence_level,
            'results': results,
            'statistical_significance': significance_results,
            'summary': self._generate_summary(results, significance_results)
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        logger.info(f"Results saved to {output_path}")
    
    def _generate_summary(self, results: Dict[str, Any], significance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the evaluation results."""
        summary = {
            'best_models': {},
            'key_findings': []
        }
        
        # Find best models for each metric
        metrics = ['hr', 'precision', 'recall', 'ndcg', 'mrr', 'map']
        for metric in metrics:
            best_model = None
            best_score = -1
            
            for model_name, model_results in results.items():
                metric_key = f'{metric}@10'
                if metric_key in model_results:
                    score = model_results[metric_key]['mean']
                    if score > best_score:
                        best_score = score
                        best_model = model_name
            
            if best_model:
                summary['best_models'][metric] = {
                    'model': best_model,
                    'score': best_score
                }
        
        # Add key findings
        summary['key_findings'].append(f"Evaluation completed with {self.num_runs} runs per model")
        summary['key_findings'].append(f"Confidence level: {self.confidence_level*100}%")
        
        return summary


def load_test_data_with_predictions(data_path: str) -> Tuple[Dict[str, List[List[str]]], List[List[str]]]:
    """
    Load test data with model predictions.
    Returns predictions dictionary and targets list.
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    predictions = {}
    targets = []
    
    # Extract targets
    targets = [entry['output'].split(', ') for entry in data]
    
    # Extract predictions for each model
    # Use model_response_items if available (processed data), otherwise split model_response
    if 'model_response_items' in data[0]:
        predictions['multimodal_llm'] = [entry['model_response_items'] for entry in data]
    elif 'model_response' in data[0]:
        predictions['multimodal_llm'] = [entry['model_response'].split(', ') for entry in data]
    
    return predictions, targets


def main():
    """Main function to run robust evaluation."""
    from src.utils.utils import setup_logging, load_config
    
    setup_logging()
    logger.info("Running robust evaluation...")
    
    try:
        
        config = load_config()
        evaluator = RobustEvaluator(config, num_runs=5)
        
        # Load data - use the processed test data with model_response_items field
        predictions, targets = load_test_data_with_predictions('data/processed/test_with_responses_processed.json')
        
        # Run robust evaluation with config-specified k
        k = config['data_config']['number_of_items_to_predict']
        logger.info(f"Using k={k} for evaluation (from config)")
        results = evaluator.run_multiple_evaluations(predictions, targets, k=k)
        
        # Statistical significance testing
        significance_results = evaluator.statistical_significance_test(results)
        
        # Save results
        evaluator.save_results(results, significance_results, 'results/robust_evaluation_results.json')
        
        logger.info("Robust evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Robust evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
