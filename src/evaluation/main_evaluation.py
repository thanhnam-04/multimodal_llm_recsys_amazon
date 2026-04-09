#!/usr/bin/env python3
"""
Main evaluation module that combines all evaluation results into a comprehensive report.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

from src.utils.utils import setup_logging, load_config

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def load_evaluation_results() -> Dict[str, Any]:
    """Load all evaluation results from different modules."""
    results_dir = Path("results")
    results = {}
    
    # Load robust evaluation results
    robust_file = results_dir / "robust_evaluation_results.json"
    if robust_file.exists():
        with open(robust_file, 'r') as f:
            results['robust_evaluation'] = json.load(f)
        logger.info(f"Loaded robust evaluation results from {robust_file}")
    else:
        logger.warning(f"Robust evaluation results not found at {robust_file}")
    
    # Load ablation study results
    ablation_file = results_dir / "ablation_study_results.json"
    if ablation_file.exists():
        with open(ablation_file, 'r') as f:
            results['ablation_studies'] = json.load(f)
        logger.info(f"Loaded ablation study results from {ablation_file}")
    else:
        logger.warning(f"Ablation study results not found at {ablation_file}")
    
    # Load cross-category evaluation results
    cross_cat_file = results_dir / "cross_category_evaluation_results.json"
    if cross_cat_file.exists():
        with open(cross_cat_file, 'r') as f:
            results['cross_category_evaluation'] = json.load(f)
        logger.info(f"Loaded cross-category evaluation results from {cross_cat_file}")
    else:
        logger.warning(f"Cross-category evaluation results not found at {cross_cat_file}")
    
    # Load baseline predictions
    baseline_file = results_dir / "test_with_baseline_predictions.json"
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            results['baseline_predictions'] = json.load(f)
        logger.info(f"Loaded baseline predictions from {baseline_file}")
    else:
        logger.warning(f"Baseline predictions not found at {baseline_file}")
    
    # Load cross-category summary
    cross_summary_file = results_dir / "cross_category_summary.json"
    if cross_summary_file.exists():
        with open(cross_summary_file, 'r') as f:
            results['cross_category_summary'] = json.load(f)
        logger.info(f"Loaded cross-category summary from {cross_summary_file}")
    else:
        logger.warning(f"Cross-category summary not found at {cross_summary_file}")
    
    return results

def extract_main_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract main metrics from all evaluation results."""
    main_metrics = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'two_track_evaluation': {},
        'model_performance': {},
        'baseline_comparison': {},
        'ablation_insights': {},
        'cross_category_performance': {},
        'summary': {}
    }
    
    # Extract robust evaluation metrics with two-track system
    if 'robust_evaluation' in results:
        robust_results = results['robust_evaluation']
        if 'results' in robust_results and 'multimodal_llm' in robust_results['results']:
            multimodal_results = robust_results['results']['multimodal_llm']
            
            # Extract two-track metrics
            main_metrics['two_track_evaluation'] = {
                'track_a_item_metrics': {
                    'item_precision_10': {
                        'mean': multimodal_results.get('item_precision@10', {}).get('mean', 0),
                        'std': multimodal_results.get('item_precision@10', {}).get('std', 0),
                        'ci_lower': multimodal_results.get('item_precision@10', {}).get('ci_lower', 0),
                        'ci_upper': multimodal_results.get('item_precision@10', {}).get('ci_upper', 0)
                    },
                    'item_recall_10': {
                        'mean': multimodal_results.get('item_recall@10', {}).get('mean', 0),
                        'std': multimodal_results.get('item_recall@10', {}).get('std', 0),
                        'ci_lower': multimodal_results.get('item_recall@10', {}).get('ci_lower', 0),
                        'ci_upper': multimodal_results.get('item_recall@10', {}).get('ci_upper', 0)
                    },
                    'item_hitrate_10': {
                        'mean': multimodal_results.get('item_hitrate@10', {}).get('mean', 0),
                        'std': multimodal_results.get('item_hitrate@10', {}).get('std', 0),
                        'ci_lower': multimodal_results.get('item_hitrate@10', {}).get('ci_lower', 0),
                        'ci_upper': multimodal_results.get('item_hitrate@10', {}).get('ci_upper', 0)
                    },
                    'item_ndcg_10': {
                        'mean': multimodal_results.get('item_ndcg@10', {}).get('mean', 0),
                        'std': multimodal_results.get('item_ndcg@10', {}).get('std', 0),
                        'ci_lower': multimodal_results.get('item_ndcg@10', {}).get('ci_lower', 0),
                        'ci_upper': multimodal_results.get('item_ndcg@10', {}).get('ci_upper', 0)
                    },
                    'item_mrr': {
                        'mean': multimodal_results.get('item_mrr', {}).get('mean', 0),
                        'std': multimodal_results.get('item_mrr', {}).get('std', 0),
                        'ci_lower': multimodal_results.get('item_mrr', {}).get('ci_lower', 0),
                        'ci_upper': multimodal_results.get('item_mrr', {}).get('ci_upper', 0)
                    },
                    'item_map': {
                        'mean': multimodal_results.get('item_map', {}).get('mean', 0),
                        'std': multimodal_results.get('item_map', {}).get('std', 0),
                        'ci_lower': multimodal_results.get('item_map', {}).get('ci_lower', 0),
                        'ci_upper': multimodal_results.get('item_map', {}).get('ci_upper', 0)
                    },
                    'item_coverage_10': {
                        'mean': multimodal_results.get('item_coverage@10', {}).get('mean', 0),
                        'std': multimodal_results.get('item_coverage@10', {}).get('std', 0),
                        'ci_lower': multimodal_results.get('item_coverage@10', {}).get('ci_lower', 0),
                        'ci_upper': multimodal_results.get('item_coverage@10', {}).get('ci_upper', 0)
                    },
                    'item_diversity_10': {
                        'mean': multimodal_results.get('item_diversity@10', {}).get('mean', 0),
                        'std': multimodal_results.get('item_diversity@10', {}).get('std', 0),
                        'ci_lower': multimodal_results.get('item_diversity@10', {}).get('ci_lower', 0),
                        'ci_upper': multimodal_results.get('item_diversity@10', {}).get('ci_upper', 0)
                    },
                    'item_novelty_10': {
                        'mean': multimodal_results.get('item_novelty@10', {}).get('mean', 0),
                        'std': multimodal_results.get('item_novelty@10', {}).get('std', 0),
                        'ci_lower': multimodal_results.get('item_novelty@10', {}).get('ci_lower', 0),
                        'ci_upper': multimodal_results.get('item_novelty@10', {}).get('ci_upper', 0)
                    }
                },
                'track_b_stop_metrics': {
                    'stop_hit_10': {
                        'mean': multimodal_results.get('stop_hit@10', {}).get('mean', 0),
                        'std': multimodal_results.get('stop_hit@10', {}).get('std', 0),
                        'ci_lower': multimodal_results.get('stop_hit@10', {}).get('ci_lower', 0),
                        'ci_upper': multimodal_results.get('stop_hit@10', {}).get('ci_upper', 0)
                    },
                    'stop_ndcg_10': {
                        'mean': multimodal_results.get('stop_ndcg@10', {}).get('mean', 0),
                        'std': multimodal_results.get('stop_ndcg@10', {}).get('std', 0),
                        'ci_lower': multimodal_results.get('stop_ndcg@10', {}).get('ci_lower', 0),
                        'ci_upper': multimodal_results.get('stop_ndcg@10', {}).get('ci_upper', 0)
                    },
                    'stop_rank_median': multimodal_results.get('stop_rank_median', None),
                    'stop_rank_mean': multimodal_results.get('stop_rank_mean', None),
                    'stop_rank_std': multimodal_results.get('stop_rank_std', None),
                    'stop_prediction_rate': multimodal_results.get('stop_prediction_rate', 0)
                }
            }
            
            # Legacy format for backward compatibility
            main_metrics['model_performance'] = {
                'hit_rate_10': {
                    'mean': multimodal_results.get('item_hitrate@10', {}).get('mean', 0),
                    'std': multimodal_results.get('item_hitrate@10', {}).get('std', 0),
                    'ci_lower': multimodal_results.get('item_hitrate@10', {}).get('ci_lower', 0),
                    'ci_upper': multimodal_results.get('item_hitrate@10', {}).get('ci_upper', 0)
                },
                'precision_10': {
                    'mean': multimodal_results.get('item_precision@10', {}).get('mean', 0),
                    'std': multimodal_results.get('item_precision@10', {}).get('std', 0),
                    'ci_lower': multimodal_results.get('item_precision@10', {}).get('ci_lower', 0),
                    'ci_upper': multimodal_results.get('item_precision@10', {}).get('ci_upper', 0)
                },
                'recall_10': {
                    'mean': multimodal_results.get('item_recall@10', {}).get('mean', 0),
                    'std': multimodal_results.get('item_recall@10', {}).get('std', 0),
                    'ci_lower': multimodal_results.get('item_recall@10', {}).get('ci_lower', 0),
                    'ci_upper': multimodal_results.get('item_recall@10', {}).get('ci_upper', 0)
                },
                'ndcg_10': {
                    'mean': multimodal_results.get('item_ndcg@10', {}).get('mean', 0),
                    'std': multimodal_results.get('item_ndcg@10', {}).get('std', 0),
                    'ci_lower': multimodal_results.get('item_ndcg@10', {}).get('ci_lower', 0),
                    'ci_upper': multimodal_results.get('item_ndcg@10', {}).get('ci_upper', 0)
                },
                'mrr': {
                    'mean': multimodal_results.get('item_mrr', {}).get('mean', 0),
                    'std': multimodal_results.get('item_mrr', {}).get('std', 0),
                    'ci_lower': multimodal_results.get('item_mrr', {}).get('ci_lower', 0),
                    'ci_upper': multimodal_results.get('item_mrr', {}).get('ci_upper', 0)
                },
                'map': {
                    'mean': multimodal_results.get('item_map', {}).get('mean', 0),
                    'std': multimodal_results.get('item_map', {}).get('std', 0),
                    'ci_lower': multimodal_results.get('item_map', {}).get('ci_lower', 0),
                    'ci_upper': multimodal_results.get('item_map', {}).get('ci_upper', 0)
                }
            }
    
    # Extract ablation study insights
    if 'ablation_studies' in results:
        ablation_results = results['ablation_studies']
        if 'summary' in ablation_results:
            main_metrics['ablation_insights'] = {
                'best_modality': ablation_results['summary'].get('best_modality', 'Unknown'),
                'best_fusion_method': ablation_results['summary'].get('best_fusion_method', 'Unknown'),
                'modality_contributions': ablation_results.get('modality_contributions', {}),
                'fusion_method_performance': ablation_results.get('fusion_method_performance', {})
            }
    
    # Extract cross-category performance
    if 'cross_category_summary' in results:
        cross_summary = results['cross_category_summary']
        main_metrics['cross_category_performance'] = {
            'category_performance': cross_summary.get('category_performance', {}),
            'best_performing_category': cross_summary.get('best_performing_category', 'Unknown'),
            'worst_performing_category': cross_summary.get('worst_performing_category', 'Unknown'),
            'performance_variance': cross_summary.get('performance_variance', 0)
        }
    
    # Generate summary insights
    main_metrics['summary'] = generate_summary_insights(main_metrics)
    
    return main_metrics

def generate_summary_insights(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Generate high-level summary insights."""
    summary = {
        'overall_performance': 'Unknown',
        'key_strengths': [],
        'key_weaknesses': [],
        'recommendations': [],
        'two_track_analysis': {}
    }
    
    # Analyze two-track evaluation if available
    if 'two_track_evaluation' in metrics and metrics['two_track_evaluation']:
        two_track = metrics['two_track_evaluation']
        
        # Track A: Item recommendation quality
        track_a = two_track.get('track_a_item_metrics', {})
        item_precision = track_a.get('item_precision_10', {}).get('mean', 0)
        item_recall = track_a.get('item_recall_10', {}).get('mean', 0)
        item_hitrate = track_a.get('item_hitrate_10', {}).get('mean', 0)
        item_ndcg = track_a.get('item_ndcg_10', {}).get('mean', 0)
        item_mrr = track_a.get('item_mrr', {}).get('mean', 0)
        
        # Track B: Stop prediction quality
        track_b = two_track.get('track_b_stop_metrics', {})
        stop_hit = track_b.get('stop_hit_10', {}).get('mean', 0)
        stop_ndcg = track_b.get('stop_ndcg_10', {}).get('mean', 0)
        stop_prediction_rate = track_b.get('stop_prediction_rate', 0)
        if isinstance(stop_prediction_rate, dict):
            stop_prediction_rate = stop_prediction_rate.get('mean', 0)
        
        # Two-track analysis
        summary['two_track_analysis'] = {
            'item_recommendation_quality': {
                'precision_score': item_precision,
                'recall_score': item_recall,
                'hit_rate_score': item_hitrate,
                'ndcg_score': item_ndcg,
                'mrr_score': item_mrr,
                'overall_rating': 'Good' if item_precision > 0.05 else 'Moderate' if item_precision > 0.01 else 'Poor'
            },
            'stop_prediction_quality': {
                'stop_hit_score': stop_hit,
                'stop_ndcg_score': stop_ndcg,
                'stop_prediction_rate': stop_prediction_rate,
                'overall_rating': 'Good' if stop_hit > 0.7 else 'Moderate' if stop_hit > 0.3 else 'Poor'
            }
        }
        
        # Overall performance based on both tracks
        if item_precision > 0.05 and stop_hit > 0.7:
            summary['overall_performance'] = 'Excellent'
        elif item_precision > 0.01 and stop_hit > 0.3:
            summary['overall_performance'] = 'Good'
        elif item_precision > 0.005 or stop_hit > 0.1:
            summary['overall_performance'] = 'Moderate'
        else:
            summary['overall_performance'] = 'Poor'
        
        # Track A strengths and weaknesses
        if item_hitrate > 0.1:
            summary['key_strengths'].append('Good item hit rate - finding relevant products')
        if item_precision > 0.05:
            summary['key_strengths'].append('High precision - few irrelevant recommendations')
        if item_recall > 0.1:
            summary['key_strengths'].append('Good recall - capturing most relevant items')
        if item_ndcg > 0.1:
            summary['key_strengths'].append('Good ranking quality - relevant items ranked highly')
        
        if item_precision < 0.01:
            summary['key_weaknesses'].append('Low precision - many irrelevant recommendations')
        if item_recall < 0.05:
            summary['key_weaknesses'].append('Low recall - missing relevant items')
        if item_hitrate < 0.05:
            summary['key_weaknesses'].append('Low hit rate - rarely finding relevant items')
        if item_mrr < 0.01:
            summary['key_weaknesses'].append('Poor ranking - relevant items ranked too low')
        
        # Track B strengths and weaknesses
        if stop_hit > 0.7:
            summary['key_strengths'].append('Excellent stop prediction - correctly predicting when to stop')
        elif stop_hit > 0.3:
            summary['key_strengths'].append('Good stop prediction - mostly correct stopping behavior')
        
        if stop_hit < 0.3:
            summary['key_weaknesses'].append('Poor stop prediction - incorrect stopping behavior')
        if stop_prediction_rate < 0.5:
            summary['key_weaknesses'].append('Low stop prediction rate - not predicting stops when needed')
        
        # Recommendations based on two-track analysis
        if summary['overall_performance'] == 'Poor':
            summary['recommendations'].extend([
                'Focus on improving item recommendation quality first',
                'Consider increasing training data size and quality',
                'Review model architecture and hyperparameters',
                'Implement better data filtering strategies',
                'Consider ensemble methods for better performance'
            ])
        elif summary['overall_performance'] == 'Moderate':
            summary['recommendations'].extend([
                'Fine-tune hyperparameters for better performance',
                'Consider data augmentation techniques',
                'Implement advanced fusion methods',
                'Focus on improving the weaker track (item vs stop prediction)'
            ])
        else:
            summary['recommendations'].extend([
                'Model performing well - consider deployment',
                'Monitor performance on new data',
                'Consider A/B testing with users',
                'Fine-tune for specific use cases'
            ])
    
    # Fallback to legacy analysis if two-track not available
    elif 'model_performance' in metrics:
        perf = metrics['model_performance']
        
        # Determine overall performance
        precision_mean = perf.get('precision_10', {}).get('mean', 0)
        recall_mean = perf.get('recall_10', {}).get('mean', 0)
        hit_rate_mean = perf.get('hit_rate_10', {}).get('mean', 0)
        
        if precision_mean > 0.05:
            summary['overall_performance'] = 'Good'
        elif precision_mean > 0.01:
            summary['overall_performance'] = 'Moderate'
        else:
            summary['overall_performance'] = 'Poor'
        
        # Identify strengths and weaknesses
        if hit_rate_mean > 0.1:
            summary['key_strengths'].append('Good hit rate performance')
        if precision_mean > 0.05:
            summary['key_strengths'].append('Reasonable precision')
        if recall_mean > 0.1:
            summary['key_strengths'].append('Good recall performance')
        
        if precision_mean < 0.01:
            summary['key_weaknesses'].append('Low precision - many irrelevant recommendations')
        if recall_mean < 0.05:
            summary['key_weaknesses'].append('Low recall - missing relevant items')
        if hit_rate_mean < 0.05:
            summary['key_weaknesses'].append('Low hit rate - rarely finding relevant items')
        
        # Add recommendations based on performance
        if summary['overall_performance'] == 'Poor':
            summary['recommendations'].extend([
                'Consider increasing training data size',
                'Review model architecture and hyperparameters',
                'Implement better data filtering strategies',
                'Consider ensemble methods'
            ])
        elif summary['overall_performance'] == 'Moderate':
            summary['recommendations'].extend([
                'Fine-tune hyperparameters',
                'Consider data augmentation',
                'Implement advanced fusion methods'
            ])
        else:
            summary['recommendations'].extend([
                'Model performing well - consider deployment',
                'Monitor performance on new data',
                'Consider A/B testing with users'
            ])
    
    return summary

def main():
    """Main function to generate comprehensive evaluation results."""
    logger.info("Starting main evaluation...")
    
    try:
        # Load all evaluation results
        results = load_evaluation_results()
        
        if not results:
            logger.error("No evaluation results found!")
            return
        
        # Extract main metrics
        main_metrics = extract_main_metrics(results)
        
        # Save main results
        output_file = Path("results/result_metrics.json")
        with open(output_file, 'w') as f:
            json.dump(main_metrics, f, indent=2)
        
        logger.info(f"Main evaluation results saved to {output_file}")
        
        # Log summary
        logger.info("=" * 60)
        logger.info("MAIN EVALUATION SUMMARY")
        logger.info("=" * 60)
        
        if 'summary' in main_metrics:
            summary = main_metrics['summary']
            logger.info(f"Overall Performance: {summary['overall_performance']}")
            
            # Display two-track analysis if available
            if 'two_track_analysis' in summary and summary['two_track_analysis']:
                two_track = summary['two_track_analysis']
                logger.info("\nTwo-Track Analysis:")
                logger.info("-" * 40)
                
                # Track A: Item recommendation quality
                track_a = two_track.get('item_recommendation_quality', {})
                logger.info(f"Track A - Item Recommendation Quality: {track_a.get('overall_rating', 'Unknown')}")
                logger.info(f"  Precision@10: {track_a.get('precision_score', 0):.4f}")
                logger.info(f"  Recall@10: {track_a.get('recall_score', 0):.4f}")
                logger.info(f"  Hit Rate@10: {track_a.get('hit_rate_score', 0):.4f}")
                logger.info(f"  NDCG@10: {track_a.get('ndcg_score', 0):.4f}")
                logger.info(f"  MRR: {track_a.get('mrr_score', 0):.4f}")
                
                # Track B: Stop prediction quality
                track_b = two_track.get('stop_prediction_quality', {})
                logger.info(f"Track B - Stop Prediction Quality: {track_b.get('overall_rating', 'Unknown')}")
                logger.info(f"  Stop Hit@10: {track_b.get('stop_hit_score', 0):.4f}")
                logger.info(f"  Stop NDCG@10: {track_b.get('stop_ndcg_score', 0):.4f}")
                logger.info(f"  Stop Prediction Rate: {track_b.get('stop_prediction_rate', 0):.4f}")
            
            if summary['key_strengths']:
                logger.info("\nKey Strengths:")
                for strength in summary['key_strengths']:
                    logger.info(f"  - {strength}")
            
            if summary['key_weaknesses']:
                logger.info("\nKey Weaknesses:")
                for weakness in summary['key_weaknesses']:
                    logger.info(f"  - {weakness}")
            
            if summary['recommendations']:
                logger.info("\nRecommendations:")
                for rec in summary['recommendations']:
                    logger.info(f"  - {rec}")
        
        logger.info("\nMain evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main evaluation: {e}")
        raise

if __name__ == "__main__":
    main()
