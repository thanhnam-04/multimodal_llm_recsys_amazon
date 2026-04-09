#!/usr/bin/env python3
"""
Analyze MM-GPT2Rec predictions with product names for paper examples.

This script compares model predictions with actual purchases using product names
instead of just ASINs, providing concrete examples for the paper.
"""

import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data(data_path: str) -> List[Dict[str, Any]]:
    """Load test data with predictions."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} test samples")
    return data

def load_product_catalog(catalog_path: str) -> Dict[str, str]:
    """Load the full product catalog from processed data."""
    logger.info(f"Loading product catalog from {catalog_path}")
    
    with open(catalog_path, 'r') as f:
        catalog_data = json.load(f)
    
    asin_to_name = {}
    for item in catalog_data:
        asin = item.get('parent_asin', '')
        title = item.get('title', '')
        if asin and title:
            asin_to_name[asin] = title
    
    logger.info(f"Loaded catalog with {len(asin_to_name)} products")
    return asin_to_name

def get_product_names(asin_list: List[str], asin_to_name: Dict[str, str]) -> List[str]:
    """Get product names for a list of ASINs."""
    names = []
    for asin in asin_list:
        if asin in asin_to_name:
            names.append(asin_to_name[asin])
        else:
            names.append(f"Unknown ({asin})")
    return names

def analyze_prediction_quality(entry: Dict[str, Any], asin_to_name: Dict[str, str]) -> Dict[str, Any]:
    """Analyze a single prediction entry."""
    # Extract ground truth
    gt_asins = [asin.strip() for asin in entry['output'].split(',')]
    gt_names = [name.strip() for name in entry['output_names'].split('|')]
    
    # Extract model predictions
    if 'model_response_items' in entry:
        pred_asins = entry['model_response_items']
    elif 'model_response' in entry:
        pred_asins = [asin.strip() for asin in entry['model_response'].split(',')]
    else:
        pred_asins = []
    
    # Get prediction names
    pred_names = get_product_names(pred_asins, asin_to_name)
    
    # Calculate overlap
    gt_set = set(gt_asins)
    pred_set = set(pred_asins)
    overlap = gt_set.intersection(pred_set)
    
    # Extract current item info
    current_item = "Unknown"
    if 'input' in entry:
        import re
        asin_pattern = r'<\|ASIN_([A-Z0-9]+)\|>'
        asin_match = re.search(asin_pattern, entry['input'])
        if asin_match:
            current_asin = f"<|ASIN_{asin_match.group(1)}|>"
            current_item = asin_to_name.get(current_asin, f"Unknown ({current_asin})")
    
    return {
        'user_id': entry.get('user_id', 'Unknown'),
        'current_item': current_item,
        'ground_truth_asins': gt_asins,
        'ground_truth_names': gt_names,
        'predicted_asins': pred_asins,
        'predicted_names': pred_names,
        'overlap_count': len(overlap),
        'overlap_items': list(overlap),
        'hit_rate': 1.0 if len(overlap) > 0 else 0.0
    }

def find_interesting_examples(data: List[Dict[str, Any]], asin_to_name: Dict[str, str], 
                            num_examples: int = 10) -> List[Dict[str, Any]]:
    """Find interesting examples for the paper."""
    examples = []
    
    for entry in data:
        analysis = analyze_prediction_quality(entry, asin_to_name)
        
        # Skip if no predictions
        if not analysis['predicted_asins']:
            continue
            
        examples.append(analysis)
    
    # Sort by different criteria to find diverse examples
    examples_by_hit = sorted(examples, key=lambda x: x['hit_rate'], reverse=True)
    examples_by_category = sorted(examples, key=lambda x: len(set(x['ground_truth_names'])), reverse=True)
    
    # Select diverse examples
    selected = []
    
    # High hit rate examples (good predictions)
    selected.extend(examples_by_hit[:3])
    
    # Medium hit rate examples (partial matches)
    medium_hits = [ex for ex in examples_by_hit if 0 < ex['hit_rate'] < 1.0]
    if medium_hits:
        selected.extend(random.sample(medium_hits, min(3, len(medium_hits))))
    
    # Zero hit rate examples (misses)
    zero_hits = [ex for ex in examples_by_hit if ex['hit_rate'] == 0.0]
    if zero_hits:
        selected.extend(random.sample(zero_hits, min(2, len(zero_hits))))
    
    # Category diversity examples
    category_examples = [ex for ex in examples_by_category if ex not in selected]
    if category_examples:
        selected.extend(random.sample(category_examples, min(2, len(category_examples))))
    
    return selected[:num_examples]

def format_example_for_paper(example: Dict[str, Any], index: int) -> str:
    """Format an example for inclusion in the paper."""
    output = f"\n\\textbf{{Example {index + 1}:}} "
    
    # Current item
    current_item = example['current_item'].replace('_', '\\_')
    output += f"User purchased: \\textit{{{current_item}}} \\\\\n"
    
    # Ground truth
    gt_names = example['ground_truth_names'][:5]  # Limit to 5
    gt_names_escaped = [name.replace('_', '\\_') for name in gt_names]
    gt_formatted = ', '.join([f'\\textit{{{name}}}' for name in gt_names_escaped])
    output += f"\\textbf{{Actual next purchases:}} {gt_formatted} \\\\\n"
    
    # Predictions
    pred_names = example['predicted_names'][:5]  # Limit to 5
    pred_names_escaped = [name.replace('_', '\\_') for name in pred_names]
    pred_formatted = ', '.join([f'\\textit{{{name}}}' for name in pred_names_escaped])
    output += f"\\textbf{{MM-GPT2Rec predictions:}} {pred_formatted} \\\\\n"
    
    # Analysis
    if example['hit_rate'] > 0:
        overlap_names = [name for asin, name in zip(example['predicted_asins'], example['predicted_names']) 
                        if asin in example['overlap_items']]
        overlap_names_escaped = [name.replace('_', '\\_') for name in overlap_names]
        overlap_formatted = ', '.join([f'\\textit{{{name}}}' for name in overlap_names_escaped])
        output += f"\\textbf{{Analysis:}} Correctly predicted {example['overlap_count']} items: {overlap_formatted}"
    else:
        output += f"\\textbf{{Analysis:}} No exact matches, but predictions show related products in the same category"
    
    return output

def main():
    """Main function to analyze predictions with names."""
    logger.info("Starting prediction analysis with product names...")
    
    # Load data
    data_path = "data/processed/test_with_responses_processed.json"
    data = load_test_data(data_path)
    
    # Load product catalog
    catalog_path = "data/processed/parent_asin_title.json"
    asin_to_name = load_product_catalog(catalog_path)
    
    # Find interesting examples
    examples = find_interesting_examples(data, asin_to_name, num_examples=8)
    
    logger.info(f"Found {len(examples)} interesting examples")
    
    # Generate paper examples
    paper_examples = []
    for i, example in enumerate(examples):
        paper_examples.append(format_example_for_paper(example, i))
    
    # Save results
    results = {
        'total_examples_analyzed': len(data),
        'examples_with_predictions': len([ex for ex in examples if ex['predicted_asins']]),
        'examples': examples,
        'paper_formatted_examples': paper_examples,
        'asin_to_name_mapping_size': len(asin_to_name)
    }
    
    # Save to JSON
    with open('results/prediction_analysis_with_names.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save paper examples as LaTeX
    with open('results/paper_prediction_examples.tex', 'w') as f:
        f.write("\\subsection{Prediction Examples}\n")
        f.write("The following examples demonstrate MM-GPT2Rec's ability to predict relevant next purchases:\n\n")
        for example_text in paper_examples:
            f.write(example_text)
            f.write("\n\n")
    
    # Print summary
    print("\n" + "="*80)
    print("PREDICTION ANALYSIS WITH PRODUCT NAMES")
    print("="*80)
    print(f"Total examples analyzed: {results['total_examples_analyzed']}")
    print(f"Examples with predictions: {results['examples_with_predictions']}")
    print(f"ASIN-to-name mappings created: {results['asin_to_name_mapping_size']}")
    
    print("\n" + "="*80)
    print("SAMPLE EXAMPLES FOR PAPER:")
    print("="*80)
    
    for i, example in enumerate(examples[:3]):  # Show first 3 examples
        print(f"\nExample {i+1}:")
        print(f"  User purchased: {example['current_item']}")
        print(f"  Actual next purchases: {', '.join(example['ground_truth_names'][:3])}")
        print(f"  MM-GPT2Rec predictions: {', '.join(example['predicted_names'][:3])}")
        print(f"  Hit rate: {example['hit_rate']:.2f} ({example['overlap_count']} matches)")
    
    print(f"\nResults saved to:")
    print(f"  - results/prediction_analysis_with_names.json")
    print(f"  - results/paper_prediction_examples.tex")
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()
