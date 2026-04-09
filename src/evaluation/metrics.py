"""
Two-track evaluation system for multimodal LLM recommendations.
Separates item recommendation quality from stopping behavior (EOS prediction).

Based on the recommendation to evaluate:
1) Item-only metrics (exclude EOS) - standard recommender metrics
2) Stop metrics (for EOS correctness) - when to stop recommending
"""

import torch
import numpy as np
import math
from typing import List, Dict, Tuple, Union
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

# Define EOS token
EOS = "<|endoftext|>"

def split_items(seq: List[str]) -> Tuple[List[str], int]:
    """
    Split sequence into items and find first EOS rank.
    
    Args:
        seq: List of items (can include EOS tokens)
        
    Returns:
        Tuple of (items_only, first_eos_rank)
        - items_only: List without EOS tokens
        - first_eos_rank: Rank (1-indexed) of first EOS, None if no EOS
    """
    items = [x for x in seq if x != EOS]
    eos_ranks = [i+1 for i, x in enumerate(seq) if x == EOS]
    first_eos_rank = eos_ranks[0] if eos_ranks else None
    return items, first_eos_rank

def precision_at_k_items(pred: List[str], truth: List[str], k: int = 10) -> float:
    """
    Calculate precision@k for items only (excluding EOS).
    
    Args:
        pred: Predicted sequence
        truth: Ground truth sequence
        k: Number of items to consider
        
    Returns:
        Precision@k for items only
    """
    p_items, _ = split_items(pred[:k])
    t_items, _ = split_items(truth[:k])
    hits = len(set(p_items) & set(t_items))
    return hits / k  # Standard precision@k definition

def recall_at_k_items(pred: List[str], truth: List[str], k: int = 10) -> float:
    """
    Calculate recall@k for items only (excluding EOS).
    
    Args:
        pred: Predicted sequence
        truth: Ground truth sequence
        k: Number of items to consider
        
    Returns:
        Recall@k for items only
    """
    p_items, _ = split_items(pred[:k])
    t_items, _ = split_items(truth[:k])
    if len(t_items) == 0:
        return 0.0  # No real items to recall
    hits = len(set(p_items) & set(t_items))
    return hits / len(t_items)

def hitrate_at_k_items(pred: List[str], truth: List[str], k: int = 10) -> float:
    """
    Calculate hit rate@k for items only (excluding EOS).
    
    Args:
        pred: Predicted sequence
        truth: Ground truth sequence
        k: Number of items to consider
        
    Returns:
        1.0 if any ground truth item appears in predictions, 0.0 otherwise
    """
    p_items, _ = split_items(pred[:k])
    t_items, _ = split_items(truth[:k])
    return 1.0 if len(set(p_items) & set(t_items)) > 0 else 0.0

def ndcg_at_k_items(pred: List[str], truth: List[str], k: int = 10) -> float:
    """
    Calculate NDCG@k for items only (excluding EOS).
    
    Args:
        pred: Predicted sequence
        truth: Ground truth sequence
        k: Number of items to consider
        
    Returns:
        NDCG@k for items only
    """
    p_items, _ = split_items(pred[:k])
    t_items, _ = split_items(truth[:k])
    
    if len(t_items) == 0:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(p_items):
        if item in t_items:
            dcg += 1.0 / math.log2(i + 2)  # +2 because log2(1) = 0
    
    # Calculate IDCG (ideal DCG)
    idcg = 0.0
    for i in range(min(len(t_items), k)):
        idcg += 1.0 / math.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0

def mrr_items(pred: List[str], truth: List[str]) -> float:
    """
    Calculate MRR for items only (excluding EOS).
    
    Args:
        pred: Predicted sequence
        truth: Ground truth sequence
        
    Returns:
        MRR for items only
    """
    p_items, _ = split_items(pred)
    t_items, _ = split_items(truth)
    
    if len(t_items) == 0:
        return 0.0
    
    for i, item in enumerate(p_items):
        if item in t_items:
            return 1.0 / (i + 1)
    return 0.0

def map_items(pred: List[str], truth: List[str]) -> float:
    """
    Calculate MAP for items only (excluding EOS).
    
    Args:
        pred: Predicted sequence
        truth: Ground truth sequence
        
    Returns:
        MAP for items only
    """
    p_items, _ = split_items(pred)
    t_items, _ = split_items(truth)
    
    if len(t_items) == 0:
        return 0.0
    
    ap = 0.0
    hits = 0
    for i, item in enumerate(p_items):
        if item in t_items:
            hits += 1
            ap += hits / (i + 1)
    
    return ap / len(t_items) if len(t_items) > 0 else 0.0

def stop_hit_at_k(pred: List[str], truth: List[str], k: int = 10) -> float:
    """
    Calculate StopHit@k - whether model correctly predicts EOS when ground truth has EOS.
    
    Args:
        pred: Predicted sequence
        truth: Ground truth sequence
        k: Number of items to consider
        
    Returns:
        1.0 if truth has EOS in next-k AND pred has EOS in top-k, 0.0 otherwise
    """
    _, t_first_eos = split_items(truth[:k])
    _, p_first_eos = split_items(pred[:k])
    return 1.0 if (t_first_eos is not None and p_first_eos is not None) else 0.0

def stop_rank(pred: List[str], k: int = 10) -> Union[int, None]:
    """
    Calculate StopRank - rank of first EOS prediction.
    
    Args:
        pred: Predicted sequence
        k: Number of items to consider
        
    Returns:
        Rank (1-indexed) of first EOS, None if no EOS predicted
    """
    _, p_first_eos = split_items(pred[:k])
    return p_first_eos

def stop_ndcg_at_k(pred: List[str], truth: List[str], k: int = 10) -> float:
    """
    Calculate Stop-nDCG@k - position-sensitive metric for EOS prediction.
    
    Args:
        pred: Predicted sequence
        truth: Ground truth sequence
        k: Number of items to consider
        
    Returns:
        Stop-nDCG@k score
    """
    _, t_first_eos = split_items(truth[:k])
    _, p_first_eos = split_items(pred[:k])
    
    if t_first_eos is None:
        return 0.0  # No stop in truth; don't reward predicting stop
    
    ideal = 1.0 / math.log2(1 + t_first_eos)
    dcg = 0.0 if p_first_eos is None else 1.0 / math.log2(1 + p_first_eos)
    
    return dcg / ideal if ideal > 0 else 0.0

def coverage_items(predictions: List[List[str]], targets: List[List[str]], k: int = 10) -> float:
    """
    Calculate coverage for items only (excluding EOS).
    
    Args:
        predictions: List of predicted sequences
        targets: List of target sequences
        k: Number of items to consider
        
    Returns:
        Coverage@k for items only
    """
    all_items = set()
    recommended_items = set()
    
    for pred in predictions:
        p_items, _ = split_items(pred[:k])
        recommended_items.update(p_items)
    
    for target in targets:
        t_items, _ = split_items(target[:k])
        all_items.update(t_items)
    
    return len(recommended_items) / len(all_items) if len(all_items) > 0 else 0.0

def diversity_items(predictions: List[List[str]], k: int = 10) -> float:
    """
    Calculate diversity for items only (excluding EOS).
    
    Args:
        predictions: List of predicted sequences
        k: Number of items to consider
        
    Returns:
        Diversity@k for items only
    """
    total_diversity = 0.0
    count = 0
    
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            p1_items, _ = split_items(predictions[i][:k])
            p2_items, _ = split_items(predictions[j][:k])
            
            pred1 = set(p1_items)
            pred2 = set(p2_items)
            
            intersection = len(pred1 & pred2)
            union = len(pred1 | pred2)
            
            if union > 0:
                total_diversity += 1 - (intersection / union)
                count += 1
    
    return total_diversity / count if count > 0 else 0.0

def novelty_items(predictions: List[List[str]], k: int = 10) -> float:
    """
    Calculate novelty for items only (excluding EOS).
    
    Args:
        predictions: List of predicted sequences
        k: Number of items to consider
        
    Returns:
        Novelty@k for items only
    """
    item_counts = {}
    
    for pred in predictions:
        p_items, _ = split_items(pred[:k])
        for item in p_items:
            item_counts[item] = item_counts.get(item, 0) + 1
    
    total_novelty = 0.0
    for pred in predictions:
        p_items, _ = split_items(pred[:k])
        pred_novelty = 0.0
        for item in p_items:
            pred_novelty += 1 / (item_counts[item] + 1)  # +1 to avoid division by zero
        total_novelty += pred_novelty / max(1, len(p_items))
    
    return total_novelty / len(predictions)

def calculate_two_track_metrics(predictions: List[List[str]], targets: List[List[str]], k: int = 10) -> Dict[str, float]:
    """
    Calculate comprehensive two-track metrics.
    
    Args:
        predictions: List of predicted sequences
        targets: List of target sequences
        k: Number of items to consider
        
    Returns:
        Dictionary with two-track metrics
    """
    metrics = {}
    
    # Track A: Item-only metrics (exclude EOS)
    metrics[f'item_precision@{k}'] = np.mean([precision_at_k_items(pred, truth, k) for pred, truth in zip(predictions, targets)])
    metrics[f'item_recall@{k}'] = np.mean([recall_at_k_items(pred, truth, k) for pred, truth in zip(predictions, targets)])
    metrics[f'item_hitrate@{k}'] = np.mean([hitrate_at_k_items(pred, truth, k) for pred, truth in zip(predictions, targets)])
    metrics[f'item_ndcg@{k}'] = np.mean([ndcg_at_k_items(pred, truth, k) for pred, truth in zip(predictions, targets)])
    metrics['item_mrr'] = np.mean([mrr_items(pred, truth) for pred, truth in zip(predictions, targets)])
    metrics['item_map'] = np.mean([map_items(pred, truth) for pred, truth in zip(predictions, targets)])
    metrics[f'item_coverage@{k}'] = coverage_items(predictions, targets, k)
    metrics[f'item_diversity@{k}'] = diversity_items(predictions, k)
    metrics[f'item_novelty@{k}'] = novelty_items(predictions, k)
    
    # Track B: Stop metrics (for EOS correctness)
    metrics[f'stop_hit@{k}'] = np.mean([stop_hit_at_k(pred, truth, k) for pred, truth in zip(predictions, targets)])
    metrics[f'stop_ndcg@{k}'] = np.mean([stop_ndcg_at_k(pred, truth, k) for pred, truth in zip(predictions, targets)])
    
    # Calculate stop rank statistics
    stop_ranks = [stop_rank(pred, k) for pred in predictions]
    stop_ranks = [r for r in stop_ranks if r is not None]  # Remove None values
    
    if stop_ranks:
        metrics['stop_rank_median'] = np.median(stop_ranks)
        metrics['stop_rank_mean'] = np.mean(stop_ranks)
        metrics['stop_rank_std'] = np.std(stop_ranks)
        metrics['stop_prediction_rate'] = len(stop_ranks) / len(predictions)
    else:
        metrics['stop_rank_median'] = None
        metrics['stop_rank_mean'] = None
        metrics['stop_rank_std'] = None
        metrics['stop_prediction_rate'] = 0.0
    
    return metrics

def calculate_metrics_v2(predictions: List[List[str]], targets: List[List[str]], k: int = 10) -> Dict[str, float]:
    """
    Main function to calculate two-track metrics.
    This replaces the old calculate_metrics function.
    
    Args:
        predictions: List of predicted sequences (strings)
        targets: List of target sequences (strings)
        k: Number of items to consider
        
    Returns:
        Dictionary with comprehensive two-track metrics
    """
    logger.info(f"Calculating two-track metrics for {len(predictions)} predictions with k={k}")
    
    # Check for empty predictions and pad with EOS tokens
    if not predictions:
        logger.warning("No predictions provided, returning zero metrics")
        return {
            f'item_precision@{k}': 0.0,
            f'item_recall@{k}': 0.0,
            f'item_hitrate@{k}': 0.0,
            f'item_ndcg@{k}': 0.0,
            'item_mrr': 0.0,
            'item_map': 0.0,
            f'item_coverage@{k}': 0.0,
            f'item_diversity@{k}': 0.0,
            f'item_novelty@{k}': 0.0,
            f'stop_hit@{k}': 0.0,
            f'stop_ndcg@{k}': 0.0,
            'stop_rank_median': None,
            'stop_rank_mean': None,
            'stop_rank_std': None,
            'stop_prediction_rate': 0.0
        }
    
    # Pad empty predictions with EOS tokens
    padded_predictions = []
    for pred in predictions:
        if not pred:  # Empty prediction
            padded_pred = [EOS] * k  # Pad with EOS tokens
            logger.debug(f"Empty prediction padded with {k} EOS tokens")
        else:
            padded_pred = pred
        padded_predictions.append(padded_pred)
    
    # Convert to string lists if needed
    if isinstance(padded_predictions[0][0], int):
        logger.warning("Converting integer predictions to strings - this may cause issues")
        padded_predictions = [[str(item) for item in pred] for pred in padded_predictions]
        targets = [[str(item) for item in target] for target in targets]
    
    return calculate_two_track_metrics(padded_predictions, targets, k)

# Backward compatibility function
def calculate_metrics(predictions: List[List[Union[str, int]]], targets: List[List[Union[str, int]]], k: int = 10) -> Dict[str, float]:
    """
    Backward compatibility wrapper for the old calculate_metrics function.
    Now uses the two-track evaluation system.
    
    Args:
        predictions: List of predicted item lists
        targets: List of target item lists
        k: Number of items to consider
        
    Returns:
        Dictionary of metric values
    """
    # Convert to string format
    str_predictions = [[str(item) for item in pred] for pred in predictions]
    str_targets = [[str(item) for item in target] for target in targets]
    
    return calculate_metrics_v2(str_predictions, str_targets, k)

if __name__ == "__main__":
    # Test the new metrics with example data
    test_predictions = [
        ["<|ASIN_B0081E9HRY|>", "<|ASIN_B07CP1KY9M|>", "<|ASIN_B07MWVCVR4|>", EOS, EOS, EOS, EOS, EOS, EOS, EOS],
        ["<|ASIN_B07FTFD1XB|>", "<|ASIN_B07CP1KY9M|>", EOS, EOS, EOS, EOS, EOS, EOS, EOS, EOS]
    ]
    
    test_targets = [
        ["<|ASIN_B07FTFD1XB|>", EOS, EOS, EOS, EOS, EOS, EOS, EOS, EOS, EOS],
        ["<|ASIN_B07FTFD1XB|>", "<|ASIN_B07CP1KY9M|>", EOS, EOS, EOS, EOS, EOS, EOS, EOS, EOS]
    ]
    
    metrics = calculate_metrics_v2(test_predictions, test_targets, k=10)
    
    print("Two-Track Evaluation Results:")
    print("=" * 50)
    print("Track A: Item Recommendation Quality (exclude EOS)")
    print(f"  Item Precision@10: {metrics['item_precision@10']:.4f}")
    print(f"  Item Recall@10: {metrics['item_recall@10']:.4f}")
    print(f"  Item Hit Rate@10: {metrics['item_hitrate@10']:.4f}")
    print(f"  Item NDCG@10: {metrics['item_ndcg@10']:.4f}")
    print(f"  Item MRR: {metrics['item_mrr']:.4f}")
    print(f"  Item MAP: {metrics['item_map']:.4f}")
    print(f"  Item Coverage@10: {metrics['item_coverage@10']:.4f}")
    print(f"  Item Diversity@10: {metrics['item_diversity@10']:.4f}")
    print(f"  Item Novelty@10: {metrics['item_novelty@10']:.4f}")
    
    print("\nTrack B: Stopping Quality")
    print(f"  Stop Hit@10: {metrics['stop_hit@10']:.4f}")
    print(f"  Stop NDCG@10: {metrics['stop_ndcg@10']:.4f}")
    print(f"  Stop Rank (median): {metrics['stop_rank_median']}")
    print(f"  Stop Rank (mean): {metrics['stop_rank_mean']:.2f}")
    print(f"  Stop Prediction Rate: {metrics['stop_prediction_rate']:.4f}")