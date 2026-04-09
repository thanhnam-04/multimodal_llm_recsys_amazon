"""
Cross-category evaluation module for comparing model performance across different product categories.
This addresses the goal of comparing model performance across various product categories.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from .metrics import calculate_metrics
from .basic_evaluation import calculate_basic_metrics

def calculate_baseline_metrics(test_users: Dict[str, List[str]], predictions: Dict[str, List[str]], k: int = 5) -> Dict[str, float]:
    """
    Calculate metrics for baseline models with dictionary format using basic evaluation.
    
    Args:
        test_users: Dict mapping user_id to list of ground truth items
        predictions: Dict mapping user_id to list of predicted items
        k: Number of items to evaluate
        
    Returns:
        Dictionary of metric values
    """
    # Convert to list format for calculate_basic_metrics
    pred_lists = []
    target_lists = []
    
    for user_id in test_users.keys():
        if user_id in predictions:
            pred_lists.append(predictions[user_id])
            target_lists.append(test_users[user_id])
        else:
            # If user not in predictions, use empty list
            pred_lists.append([])
            target_lists.append(test_users[user_id])
    
    # Get all unique items for coverage, diversity, novelty
    all_items = set()
    for items in target_lists:
        all_items.update(items)
    for items in pred_lists:
        all_items.update(items)
    
    # Remove EOS tokens from all_items
    all_items = [item for item in all_items if item != "<|endoftext|>"]
    
    return calculate_basic_metrics(pred_lists, target_lists, k=k)
from ..baselines.recommender import (
    CollaborativeFiltering,
    ContentBasedRecommender, 
    HybridRecommender,
    PopularityRecommender,
    MatrixFactorization
)

logger = logging.getLogger(__name__)

class CrossCategoryEvaluator:
    """
    Evaluator for comparing model performance across different product categories.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Use categories with sufficient data (more than 100 samples)
        self.categories = ['Tools & Home Improvement', 'Appliances', 'Amazon Home', 'Industrial & Scientific']
        self.k = config['data_config']['number_of_items_to_predict']  # Use config value instead of hardcoded 10
        self.results = {}
        
    def load_category_data(self, category: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Load train/val/test data for a specific category.
        
        Args:
            category: Category name
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        data_dir = Path("data/processed")
        
        # Try to load category-specific data first
        category_test_file = data_dir / "category_tests" / f"{category}_test.json"
        
        if category_test_file.exists():
            logger.info(f"Loading category-specific test data for {category}")
            with open(category_test_file, 'r') as f:
                test_data = json.load(f)
            
            # For train/val, we'll use the main splits filtered by category
            with open(data_dir / "train.json", 'r') as f:
                all_train = json.load(f)
            with open(data_dir / "val.json", 'r') as f:
                all_val = json.load(f)
            
            train_data = [item for item in all_train if item.get('main_category') == category]
            val_data = [item for item in all_val if item.get('main_category') == category]
            
        else:
            # Fallback: load main splits and filter by category
            logger.info(f"Loading filtered data for {category}")
            
            with open(data_dir / "train.json", 'r') as f:
                all_train = json.load(f)
            with open(data_dir / "val.json", 'r') as f:
                all_val = json.load(f)
            with open(data_dir / "test.json", 'r') as f:
                all_test = json.load(f)
            
            train_data = [item for item in all_train if item.get('main_category') == category]
            val_data = [item for item in all_val if item.get('main_category') == category]
            test_data = [item for item in all_test if item.get('main_category') == category]
        
        logger.info(f"Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples for {category}")
        return train_data, val_data, test_data
    
    def evaluate_category_performance(self, category: str, model_predictions: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Evaluate model performance on a specific category.
        
        Args:
            category: Category name
            model_predictions: Dictionary mapping user_id to list of recommended items
            
        Returns:
            Dictionary of metrics for the category
        """
        logger.info(f"Evaluating performance for category: {category}")
        
        # Load test data for this category
        _, _, test_data = self.load_category_data(category)
        
        if not test_data:
            logger.warning(f"No test data found for category {category}")
            return {}
        
        # Prepare test data for evaluation
        test_users = {}
        for item in test_data:
            user_id = item['user_id']
            if user_id not in test_users:
                test_users[user_id] = []
            test_users[user_id].append(item['parent_asin'])
        
        # Calculate metrics using config k value
        metrics = calculate_baseline_metrics(test_users, model_predictions, k=self.k)
        
        logger.info(f"Category {category} metrics: {metrics}")
        return metrics
    
    def evaluate_baselines_on_category(self, category: str) -> Dict[str, Dict[str, float]]:
        """
        Evaluate baseline methods on a specific category.
        
        Args:
            category: Category name
            
        Returns:
            Dictionary mapping baseline name to metrics
        """
        logger.info(f"Evaluating baselines for category: {category}")
        
        train_data, val_data, test_data = self.load_category_data(category)
        
        # Check if we have sufficient data (at least 10 samples each)
        min_samples = 10
        if len(train_data) < min_samples or len(test_data) < min_samples:
            logger.warning(f"Insufficient data for category {category}: {len(train_data)} train, {len(test_data)} test samples")
            return {}
        
        logger.info(f"Evaluating {category} with {len(train_data)} train and {len(test_data)} test samples")
        
        baseline_results = {}
        
        # Prepare data for baselines - convert to DataFrame format
        train_df = pd.DataFrame(train_data)
        test_users = {}
        for item in test_data:
            user_id = item['user_id']
            if user_id not in test_users:
                test_users[user_id] = []
            test_users[user_id].append(item['parent_asin'])
        
        # Evaluate Collaborative Filtering
        try:
            cf_recommender = CollaborativeFiltering()
            cf_recommender.fit(train_df)
            cf_predictions = {}
            for user_id in test_users.keys():
                pred = cf_recommender.predict(user_id, n_items=10)
                cf_predictions[user_id] = pred
            baseline_results['Collaborative_Filtering'] = calculate_baseline_metrics(test_users, cf_predictions, k=self.k)
        except Exception as e:
            logger.error(f"CF evaluation failed for {category}: {e}")
            baseline_results['Collaborative_Filtering'] = {}
        
        # Evaluate Content-Based
        try:
            cb_recommender = ContentBasedRecommender()
            cb_recommender.fit(train_df)
            cb_predictions = {}
            for user_id in test_users.keys():
                cb_predictions[user_id] = cb_recommender.predict(user_id, n_items=10)
            baseline_results['Content_Based'] = calculate_baseline_metrics(test_users, cb_predictions, k=self.k)
        except Exception as e:
            logger.error(f"CB evaluation failed for {category}: {e}")
            baseline_results['Content_Based'] = {}
        
        # Evaluate Popularity
        try:
            pop_recommender = PopularityRecommender()
            pop_recommender.fit(train_df)
            # Popularity recommender returns same recommendations for all users
            popular_items = pop_recommender.predict(n_items=10)
            pop_predictions = {}
            for user_id in test_users.keys():
                pop_predictions[user_id] = popular_items
            baseline_results['Popularity'] = calculate_baseline_metrics(test_users, pop_predictions, k=self.k)
        except Exception as e:
            logger.error(f"Popularity evaluation failed for {category}: {e}")
            baseline_results['Popularity'] = {}
        
        # Evaluate Matrix Factorization
        try:
            mf_recommender = MatrixFactorization()
            mf_recommender.fit(train_df)
            mf_predictions = {}
            for user_id in test_users.keys():
                mf_predictions[user_id] = mf_recommender.predict(user_id, n_items=10)
            baseline_results['Matrix_Factorization'] = calculate_baseline_metrics(test_users, mf_predictions, k=self.k)
        except Exception as e:
            logger.error(f"MF evaluation failed for {category}: {e}")
            baseline_results['Matrix_Factorization'] = {}
        
        return baseline_results
    
    def run_cross_category_evaluation(self, model_predictions: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive cross-category evaluation.
        
        Args:
            model_predictions: Model predictions (if None, will evaluate baselines only)
            
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info("Starting cross-category evaluation...")
        
        all_results = {
            'category_performance': {},
            'baseline_performance': {},
            'comparison_analysis': {},
            'statistical_tests': {}
        }
        
        # Evaluate each category
        for category in self.categories:
            logger.info(f"Evaluating category: {category}")
            
            # Evaluate model performance (if predictions provided)
            if model_predictions:
                model_metrics = self.evaluate_category_performance(category, model_predictions)
                all_results['category_performance'][category] = model_metrics
            
            # Evaluate baselines
            baseline_metrics = self.evaluate_baselines_on_category(category)
            all_results['baseline_performance'][category] = baseline_metrics
        
        # Perform comparison analysis
        all_results['comparison_analysis'] = self.analyze_category_differences(all_results)
        
        # Save results
        self.save_results(all_results)
        
        logger.info("Cross-category evaluation completed!")
        return all_results
    
    def analyze_category_differences(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze performance differences across categories.
        
        Args:
            results: Evaluation results
            
        Returns:
            Analysis results
        """
        logger.info("Analyzing category performance differences...")
        
        analysis = {
            'best_performing_category': {},
            'worst_performing_category': {},
            'category_rankings': {},
            'performance_variance': {},
            'category_insights': {}
        }
        
        # Analyze model performance across categories
        if 'category_performance' in results and results['category_performance']:
            model_metrics = results['category_performance']
            
            # Find best/worst performing categories for each metric
            for metric in [f'HR@{self.k}', 'MRR', f'NDCG@{self.k}', f'Precision@{self.k}']:
                if all(metric in cat_metrics for cat_metrics in model_metrics.values()):
                    metric_values = {cat: metrics[metric] for cat, metrics in model_metrics.items()}
                    
                    best_cat = max(metric_values, key=metric_values.get)
                    worst_cat = min(metric_values, key=metric_values.get)
                    
                    analysis['best_performing_category'][metric] = {
                        'category': best_cat,
                        'value': metric_values[best_cat]
                    }
                    analysis['worst_performing_category'][metric] = {
                        'category': worst_cat,
                        'value': metric_values[worst_cat]
                    }
                    
                    # Calculate variance
                    values = list(metric_values.values())
                    analysis['performance_variance'][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'range': np.max(values) - np.min(values)
                    }
        
        # Analyze baseline performance
        if 'baseline_performance' in results:
            baseline_metrics = results['baseline_performance']
            
            # Find best baseline for each category
            for category, baselines in baseline_metrics.items():
                if baselines:
                    best_baseline = {}
                    for metric in [f'HR@{self.k}', 'MRR', f'NDCG@{self.k}']:
                        metric_values = {name: metrics.get(metric, 0) for name, metrics in baselines.items()}
                        if metric_values:
                            best_baseline[metric] = max(metric_values, key=metric_values.get)
                    
                    analysis['category_insights'][category] = {
                        'best_baseline': best_baseline,
                        'num_baselines_tested': len(baselines)
                    }
        
        return analysis
    
    def create_performance_visualizations(self, results: Dict[str, Any], output_dir: str = "output/plots"):
        """
        Create visualizations for cross-category performance analysis.
        
        Args:
            results: Evaluation results
            output_dir: Output directory for plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Creating cross-category performance visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Category Performance Comparison
        if 'category_performance' in results and results['category_performance']:
            self._plot_category_comparison(results['category_performance'], output_path)
        
        # 2. Baseline Performance Across Categories
        if 'baseline_performance' in results:
            self._plot_baseline_comparison(results['baseline_performance'], output_path)
        
        # 3. Performance Variance Analysis
        if 'comparison_analysis' in results and 'performance_variance' in results['comparison_analysis']:
            self._plot_performance_variance(results['comparison_analysis']['performance_variance'], output_path)
        
        logger.info(f"Visualizations saved to {output_path}")
    
    def _plot_category_comparison(self, category_performance: Dict[str, Dict[str, float]], output_path: Path):
        """Plot category performance comparison."""
        metrics = [f'HR@{self.k}', 'MRR', f'NDCG@{self.k}', f'Precision@{self.k}']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            categories = list(category_performance.keys())
            values = [category_performance[cat].get(metric, 0) for cat in categories]
            
            bars = axes[i].bar(categories, values, color=plt.cm.Set3(np.linspace(0, 1, len(categories))))
            axes[i].set_title(f'{metric} Across Categories')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'category_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_baseline_comparison(self, baseline_performance: Dict[str, Dict[str, Dict[str, float]]], output_path: Path):
        """Plot baseline performance across categories."""
        metrics = [f'HR@{self.k}', 'MRR', f'NDCG@{self.k}']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(metrics):
            categories = list(baseline_performance.keys())
            baselines = set()
            for cat_data in baseline_performance.values():
                baselines.update(cat_data.keys())
            
            x = np.arange(len(categories))
            width = 0.8 / max(len(baselines), 1)  # Avoid division by zero
            
            if len(baselines) == 0:
                axes[i].text(0.5, 0.5, 'No baseline data available', 
                           ha='center', va='center', transform=axes[i].transAxes)
                continue
                
            for j, baseline in enumerate(baselines):
                values = [baseline_performance[cat].get(baseline, {}).get(metric, 0) for cat in categories]
                axes[i].bar(x + j * width, values, width, label=baseline)
            
            axes[i].set_title(f'{metric} - Baseline Comparison')
            axes[i].set_ylabel(metric)
            axes[i].set_xlabel('Category')
            axes[i].set_xticks(x + width * (len(baselines) - 1) / 2)
            axes[i].set_xticklabels(categories, rotation=45)
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'baseline_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_variance(self, performance_variance: Dict[str, Dict[str, float]], output_path: Path):
        """Plot performance variance analysis."""
        metrics = list(performance_variance.keys())
        means = [performance_variance[metric]['mean'] for metric in metrics]
        stds = [performance_variance[metric]['std'] for metric in metrics]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(metrics, means, yerr=stds, capsize=5, color=plt.cm.Set2(np.linspace(0, 1, len(metrics))))
        ax.set_title('Performance Variance Across Categories')
        ax.set_ylabel('Metric Value')
        ax.set_xlabel('Metrics')
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                   f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_variance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "results"):
        """
        Save cross-category evaluation results.
        
        Args:
            results: Evaluation results
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        with open(output_path / 'cross_category_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary report
        summary = self._create_summary_report(results)
        with open(output_path / 'cross_category_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Cross-category evaluation results saved to {output_path}")
    
    def _create_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary report of the evaluation results."""
        summary = {
            'evaluation_summary': {
                'categories_evaluated': self.categories,
                'total_categories': len(self.categories),
                'evaluation_date': pd.Timestamp.now().isoformat()
            },
            'key_findings': {},
            'recommendations': []
        }
        
        # Extract key findings
        if 'comparison_analysis' in results:
            analysis = results['comparison_analysis']
            
            if 'best_performing_category' in analysis:
                summary['key_findings']['best_categories'] = analysis['best_performing_category']
            
            if 'performance_variance' in analysis:
                summary['key_findings']['performance_variance'] = analysis['performance_variance']
        
        # Generate recommendations
        if 'performance_variance' in summary['key_findings']:
            variance_data = summary['key_findings']['performance_variance']
            
            for metric, stats in variance_data.items():
                if stats['std'] > 0.05:  # High variance threshold
                    summary['recommendations'].append(
                        f"High variance in {metric} across categories (std={stats['std']:.3f}). "
                        f"Consider category-specific model tuning."
                    )
        
        return summary


def run_cross_category_evaluation(config: Dict[str, Any], model_predictions: Dict[str, List[str]] = None) -> Dict[str, Any]:
    """
    Convenience function to run cross-category evaluation.
    
    Args:
        config: Configuration dictionary
        model_predictions: Model predictions (optional)
        
    Returns:
        Evaluation results
    """
    evaluator = CrossCategoryEvaluator(config)
    results = evaluator.run_cross_category_evaluation(model_predictions)
    
    # Create visualizations
    evaluator.create_performance_visualizations(results)
    
    return results


def main():
    """Main function to run cross-category evaluation."""
    from src.utils.utils import setup_logging, load_config
    
    setup_logging()
    logger.info("Running cross-category evaluation...")
    
    try:
        config = load_config()
        
        # Run cross-category evaluation
        results = run_cross_category_evaluation(config)
        
        logger.info("Cross-category evaluation completed!")
        logger.info(f"Categories evaluated: {list(results['category_performance'].keys())}")
        logger.info(f"Results saved to: results/cross_category_evaluation_results.json")
        
    except Exception as e:
        logger.error(f"Cross-category evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()


