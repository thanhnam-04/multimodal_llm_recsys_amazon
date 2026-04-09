"""
Basic evaluation script for multimodal LLM recommendations.
Calculates standard recommendation metrics: Hit Rate, Precision, NDCG, MRR, MAP, Coverage, Diversity, Novelty.
"""

import json
import numpy as np
import logging
from typing import List, Dict, Tuple, Set
from pathlib import Path
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

def precision_at_k(predictions: List[str], targets: List[str], k: int) -> float:
    """Calculate Precision@k."""
    if not predictions or not targets:
        return 0.0
    
    # Take first k predictions
    pred_k = predictions[:k]
    target_set = set(targets)

    relevant_count = sum(1 for item in pred_k if item in target_set)
    
    return relevant_count / k if k > 0 else 0.0

def recall_at_k(predictions: List[str], targets: List[str], k: int) -> float:
    """Calculate Recall@k."""
    if not predictions or not targets:
        return 0.0
    
    pred_k = predictions[:k]
    target_set = set(targets)
    
    # Count relevant items in top-k
    relevant_count = sum(1 for item in pred_k if item in target_set)
    
    return relevant_count / len(target_set) if len(target_set) > 0 else 0.0

def hit_rate_at_k(predictions: List[str], targets: List[str], k: int) -> float:
    """Calculate Hit Rate@k."""
    if not predictions or not targets:
        return 0.0
    
    pred_k = predictions[:k]
    target_set = set(targets)
    
    # Check if any relevant item is in top-k
    return 1.0 if any(item in target_set for item in pred_k) else 0.0

def ndcg_at_k(predictions: List[str], targets: List[str], k: int) -> float:
    """Calculate NDCG@k."""
    if not predictions or not targets:
        return 0.0
    
    pred_k = predictions[:k]
    target_set = set(targets)
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(pred_k):
        if item in target_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate IDCG (ideal DCG)
    idcg = 0.0
    for i in range(min(len(target_set), k)):
        idcg += 1.0 / math.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0

def mrr(predictions: List[str], targets: List[str]) -> float:
    """Calculate Mean Reciprocal Rank."""
    if not predictions or not targets:
        return 0.0
    
    target_set = set(targets)
    
    # Find rank of first relevant item
    for i, item in enumerate(predictions):
        if item in target_set:
            return 1.0 / (i + 1)  # i+1 because rank is 1-indexed
    
    return 0.0

def map_at_k(predictions: List[str], targets: List[str], k: int) -> float:
    """Calculate MAP@k."""
    if not predictions or not targets:
        return 0.0
    
    pred_k = predictions[:k]
    target_set = set(targets)
    
    # Calculate average precision
    relevant_count = 0
    precision_sum = 0.0
    
    for i, item in enumerate(pred_k):
        if item in target_set:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    
    return precision_sum / len(target_set) if len(target_set) > 0 else 0.0

def coverage(predictions: List[List[str]], all_items: Set[str]) -> float:
    """Calculate Coverage: fraction of all items that appear in recommendations."""
    if not predictions or not all_items:
        return 0.0
    
    recommended_items = set()
    for pred_list in predictions:
        recommended_items.update(pred_list)
    
    return len(recommended_items) / len(all_items)

def diversity(predictions: List[List[str]]) -> float:
    """Calculate Diversity: average pairwise Jaccard distance between recommendation lists."""
    if len(predictions) < 2:
        return 0.0
    
    total_distance = 0.0
    count = 0
    
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            set_i = set(predictions[i])
            set_j = set(predictions[j])
            
            # Jaccard distance = 1 - Jaccard similarity
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            
            if union > 0:
                jaccard_similarity = intersection / union
                jaccard_distance = 1.0 - jaccard_similarity
                total_distance += jaccard_distance
                count += 1
    
    return total_distance / count if count > 0 else 0.0

def novelty(predictions: List[List[str]], item_popularity: Dict[str, int]) -> float:
    """Calculate Novelty: average -log2(popularity) of recommended items."""
    if not predictions or not item_popularity:
        return 0.0
    
    total_novelty = 0.0
    total_items = 0
    
    for pred_list in predictions:
        for item in pred_list:
            if item in item_popularity:
                popularity = item_popularity[item]
                if popularity > 0:
                    novelty_score = -math.log2(popularity)
                    total_novelty += novelty_score
                    total_items += 1
    
    return total_novelty / total_items if total_items > 0 else 0.0

def calculate_basic_metrics(predictions: List[List[str]], targets: List[List[str]], k: int = 5) -> Dict[str, float]:
    """
    Calculate basic recommendation metrics.
    
    Args:
        predictions: List of predicted item lists
        targets: List of ground truth item lists
        k: Number of items to consider for @k metrics
        
    Returns:
        Dictionary of metric values
    """
    if not predictions or not targets:
        logger.warning("Empty predictions or targets")
        return {
            f'precision@{k}': 0.0,
            f'recall@{k}': 0.0,
            f'hit_rate@{k}': 0.0,
            f'ndcg@{k}': 0.0,
            'mrr': 0.0,
            f'map@{k}': 0.0,
            'coverage': 0.0,
            'diversity': 0.0,
            'novelty': 0.0
        }
    
    # Calculate item-level metrics
    precision_scores = []
    recall_scores = []
    hit_rate_scores = []
    ndcg_scores = []
    mrr_scores = []
    map_scores = []
    
    for pred, target in zip(predictions, targets):
        precision_scores.append(precision_at_k(pred, target, k))
        recall_scores.append(recall_at_k(pred, target, k))
        hit_rate_scores.append(hit_rate_at_k(pred, target, k))
        ndcg_scores.append(ndcg_at_k(pred, target, k))
        mrr_scores.append(mrr(pred, target))
        map_scores.append(map_at_k(pred, target, k))
    
    # Calculate system-level metrics
    all_items = set()
    for target_list in targets:
        all_items.update(target_list)
    
    # Calculate item popularity for novelty
    item_popularity = defaultdict(int)
    for target_list in targets:
        for item in target_list:
            item_popularity[item] += 1
    
    # Calculate metrics
    metrics = {
        f'precision@{k}': np.mean(precision_scores),
        f'recall@{k}': np.mean(recall_scores),
        f'hit_rate@{k}': np.mean(hit_rate_scores),
        f'ndcg@{k}': np.mean(ndcg_scores),
        'mrr': np.mean(mrr_scores),
        f'map@{k}': np.mean(map_scores),
        'coverage': coverage(predictions, all_items),
        'diversity': diversity(predictions),
        'novelty': novelty(predictions, item_popularity)
    }
    
    return metrics

def load_test_data_with_predictions(data_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Load test data with model predictions.
    
    Args:
        data_path: Path to the test data file
        
    Returns:
        Tuple of (predictions, targets)
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    predictions = []
    targets = []
    
    for entry in data:
        # Extract targets
        target_items = entry['output'].split(', ')
        targets.append(target_items)
        
        # Extract predictions
        if 'model_response_items' in entry:
            # Use processed data
            predictions.append(entry['model_response_items'])
        elif 'model_response' in entry:
            # Use raw data
            pred_items = entry['model_response'].split(', ')
            predictions.append(pred_items)
        else:
            logger.warning(f"No model response found for entry: {entry.get('user_id', 'unknown')}")
            predictions.append([])
    
    return predictions, targets

def main():
    """Main function to run basic evaluation."""
    import argparse
    from src.utils.utils import setup_logging, load_config
    
    parser = argparse.ArgumentParser(description='Basic evaluation for multimodal LLM')
    parser.add_argument('--data_path', type=str, default='data/processed/test_with_responses_processed.json',
                       help='Path to test data with predictions')
    parser.add_argument('--k', type=int, default=None,
                       help='Number of items to consider (default: from config)')
    parser.add_argument('--output_path', type=str, default='results/basic_evaluation_results.json',
                       help='Path to save results')
    
    args = parser.parse_args()
    
    setup_logging()
    logger.info("Running basic evaluation...")
    
    try:
        # Load config to get k value
        config = load_config()
        k = args.k if args.k is not None else config['data_config']['number_of_items_to_predict']
        
        logger.info(f"Using k={k} for evaluation")
        
        # Load data with fallback if processed predictions file is missing.
        data_path = Path(args.data_path)
        if not data_path.exists():
            fallback_path = Path('data/processed/test_with_responses.json')
            if fallback_path.exists():
                logger.warning(
                    f"Data path not found: {data_path}. Falling back to {fallback_path}."
                )
                data_path = fallback_path
            else:
                raise FileNotFoundError(
                    f"Neither {data_path} nor {fallback_path} exists."
                )

        predictions, targets = load_test_data_with_predictions(str(data_path))
        
        logger.info(f"Loaded {len(predictions)} predictions and {len(targets)} targets")
        
        # Calculate metrics
        metrics = calculate_basic_metrics(predictions, targets, k=k)
        
        # Log results
        logger.info("Basic Evaluation Results:")
        logger.info("=" * 50)
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        # Save results
        results = {
            'evaluation_type': 'basic',
            'k': k,
            'num_samples': len(predictions),
            'metrics': metrics,
            'data_path': str(data_path)
        }
        
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Results saved to {args.output_path}")
        logger.info("Basic evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Basic evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
