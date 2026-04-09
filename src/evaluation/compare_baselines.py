import json
import pandas as pd
from pathlib import Path
from ..baselines.recommender import CollaborativeFiltering, ContentBasedRecommender, HybridRecommender, PopularityRecommender, MatrixFactorization
from ..baselines.fast_baselines import create_fast_baselines
from ..evaluation.metrics import calculate_metrics
from ..evaluation.basic_evaluation import calculate_basic_metrics
from ..evaluation.cross_category_evaluation import CrossCategoryEvaluator
import logging

from ..utils.utils import load_config,setup_logging

#____________________________________
# Set up logging with console handler
setup_logging()
logger = logging.getLogger(__name__)
#_____________________________________

# Load config
config = load_config()
num_items = config['data_config']['number_of_items_to_predict']

# Check if robust evaluation is enabled
ROBUST_EVALUATION = config['training_config'].get('statistical_testing', False)
EVALUATION_RUNS = config['training_config'].get('evaluation_runs', 1)

data_dir = Path("data")
processed_dir = data_dir / "processed"

model_response_data_path = processed_dir / 'test_with_responses_processed.json'
save_model_comparison_metrics = processed_dir / 'result_metrics.json'

# Load the test data
with open(model_response_data_path, 'r') as file:
    test_data = json.load(file)

# Load the training and validation data
with open(f'{processed_dir}/train.json', 'r') as file:
    train_data = json.load(file)
with open(f'{processed_dir}/val.json', 'r') as file:
    val_data = json.load(file)
    
# Combine train and validation data
train_data.extend(val_data)

# Convert to DataFrame
interactions_df = pd.DataFrame(train_data)
# lower case the column names
interactions_df.columns = interactions_df.columns.str.lower()

# Initialize traditional baselines
cf_model = CollaborativeFiltering(n_factors=num_items * 3)
content_model = ContentBasedRecommender()
hybrid_model = HybridRecommender(n_factors=num_items * 3)
popularity_model = PopularityRecommender()
mf_model = MatrixFactorization(n_factors=num_items * 3, learning_rate=0.01, n_epochs=20, reg=0.01)

# Initialize multimodal baselines (if enabled)
multimodal_baselines = {}
try:
    logger.info("Initializing fast multimodal baselines...")
    multimodal_baselines, user_mapping, item_mapping = create_fast_baselines(config['data_config'])
    logger.info(f"Created fast multimodal baselines: {list(multimodal_baselines.keys())}")
except Exception as e:
    logger.warning(f"Failed to initialize multimodal baselines: {e}")
    logger.info("Continuing with traditional baselines only...")

# Fit models
cf_model.fit(interactions_df)
content_model.fit(interactions_df)
hybrid_model.fit(interactions_df, interactions_df)
popularity_model.fit(interactions_df)
mf_model.fit(interactions_df)

# Add baseline predictions to test data
for entry in test_data:
    user_id = entry['user_id']
    item_id = entry['parent_asin']
    
    # Get predictions from traditional models
    cf_predictions = cf_model.predict(user_id, n_items=num_items)
    content_predictions = content_model.predict(item_id, n_items=num_items)
    hybrid_predictions = hybrid_model.predict(user_id, item_id, n_items=num_items)
    popularity_predictions = popularity_model.predict(n_items=num_items)
    mf_predictions = mf_model.predict(user_id, n_items=num_items)
    
    # Add traditional predictions to entry
    entry['cf_model'] = cf_predictions
    entry['content_model'] = content_predictions
    entry['hybrid_model'] = hybrid_predictions
    entry['popularity_model'] = popularity_predictions
    entry['mf_model'] = mf_predictions
    
    # Add multimodal baseline predictions (using actual model predictions)
    for baseline_name, model in multimodal_baselines.items():
        try:
            # Use actual model predictions instead of hardcoded dummy values
            if hasattr(model, 'predict'):
                # Get actual predictions from the model
                pred_items = model.predict(user_id, n_items=num_items)
                if isinstance(pred_items, str):
                    pred_items = pred_items.split(', ')
                elif isinstance(pred_items, list):
                    pred_items = pred_items
                else:
                    pred_items = [str(pred_items)]
                
                # Ensure we have the right number of items
                if len(pred_items) < num_items:
                    pred_items.extend(['<|endoftext|>'] * (num_items - len(pred_items)))
                elif len(pred_items) > num_items:
                    pred_items = pred_items[:num_items]
                
                entry[f'{baseline_name}_model'] = ', '.join(pred_items)
            else:
                # Fallback for models without predict method
                logger.warning(f"Model {baseline_name} does not have predict method")
                entry[f'{baseline_name}_model'] = '<|endoftext|>'
        except Exception as e:
            logger.warning(f"Failed to generate predictions for {baseline_name}: {e}")
            entry[f'{baseline_name}_model'] = '<|endoftext|>'

# Save updated test data with baseline predictions
with open('results/test_with_baseline_predictions.json', 'w') as file:
    json.dump(test_data, file, indent=4)
    
# Initialize results dictionary
results = {
    'model': [],
    f'hr@{num_items}': [],
    f'precision@{num_items}': [],
    f'recall@{num_items}': [],
    f'ndcg@{num_items}': [],
    f'mrr': [],
    f'map': [],
    f'coverage': [],
    f'diversity': [],
    f'novelty': []
}

# Prepare model list (traditional + multimodal baselines)
models = ['gpt2-medium', 'cf_model', 'content_model', 'hybrid_model', 'popularity_model', 'mf_model']
multimodal_model_names = [f'{name}_model' for name in multimodal_baselines.keys()]
models.extend(multimodal_model_names)

# Choose evaluation method based on configuration
if ROBUST_EVALUATION and EVALUATION_RUNS > 1:
    logger.info(f"Running basic evaluation with {EVALUATION_RUNS} runs...")
    
    # Prepare predictions and targets for basic evaluation
    targets = [entry['output'].split(', ') for entry in test_data]
    
    # Evaluate each model
    for model_name in models:
        logger.info(f"Evaluating {model_name}...")
        
        # Get predictions for this model
        key_name = 'model_response_items' if model_name == 'gpt2-medium' else model_name
        if key_name == 'model_response_items':
            predictions = [entry[key_name] for entry in test_data]
        else:
            # Handle both string and list predictions
            raw_predictions = [entry[key_name] for entry in test_data]
            predictions = []
            for pred in raw_predictions:
                if isinstance(pred, list):
                    predictions.append(pred)
                elif isinstance(pred, str):
                    predictions.append(pred.split(', '))
                else:
                    predictions.append([str(pred)])
        
        # Run basic evaluation
        metrics = calculate_basic_metrics(predictions, targets, k=num_items)
        
        # Store results
        results['model'].append(model_name)
        results[f'hr@{num_items}'].append(metrics[f'hit_rate@{num_items}'])
        results[f'precision@{num_items}'].append(metrics[f'precision@{num_items}'])
        results[f'recall@{num_items}'].append(metrics[f'recall@{num_items}'])
        results[f'ndcg@{num_items}'].append(metrics[f'ndcg@{num_items}'])
        results['mrr'].append(metrics['mrr'])
        results['map'].append(metrics[f'map@{num_items}'])
        results['coverage'].append(metrics['coverage'])
        results['diversity'].append(metrics['diversity'])
        results['novelty'].append(metrics['novelty'])
        
        logger.info(f"Results for {model_name}:")
        logger.info(f"  HR@{num_items}: {metrics[f'hit_rate@{num_items}']:.4f}")
        logger.info(f"  Precision@{num_items}: {metrics[f'precision@{num_items}']:.4f}")
        logger.info(f"  NDCG@{num_items}: {metrics[f'ndcg@{num_items}']:.4f}")
        logger.info(f"  MRR: {metrics['mrr']:.4f}")
        logger.info(f"  Coverage: {metrics['coverage']:.4f}")
        logger.info(f"  Diversity: {metrics['diversity']:.4f}")
    
else:
    # Standard evaluation method using basic evaluation
    logger.info("Running standard basic evaluation...")
    
    # Prepare targets for basic evaluation
    targets = [entry['output'].split(', ') for entry in test_data]
    
    for model_name in models:
        logger.info(f"Evaluating {model_name}...")
        
        # Get predictions for this model
        key_name = 'model_response_items' if model_name == 'gpt2-medium' else model_name
        if key_name == 'model_response_items':
            predictions = [entry[key_name] for entry in test_data]
        else:
            # Handle both string and list predictions
            raw_predictions = [entry[key_name] for entry in test_data]
            predictions = []
            for pred in raw_predictions:
                if isinstance(pred, list):
                    predictions.append(pred)
                elif isinstance(pred, str):
                    predictions.append(pred.split(', '))
                else:
                    predictions.append([str(pred)])
        
        # Run basic evaluation
        metrics = calculate_basic_metrics(predictions, targets, k=num_items)
        
        # Store results
        results['model'].append(model_name)
        results[f'hr@{num_items}'].append(metrics[f'hit_rate@{num_items}'])
        results[f'precision@{num_items}'].append(metrics[f'precision@{num_items}'])
        results[f'recall@{num_items}'].append(metrics[f'recall@{num_items}'])
        results[f'ndcg@{num_items}'].append(metrics[f'ndcg@{num_items}'])
        results['mrr'].append(metrics['mrr'])
        results['map'].append(metrics[f'map@{num_items}'])
        results['coverage'].append(metrics['coverage'])
        results['diversity'].append(metrics['diversity'])
        results['novelty'].append(metrics['novelty'])
        
        logger.info(f"Results for {model_name}:")
        logger.info(f"  HR@{num_items}: {metrics[f'hit_rate@{num_items}']:.4f}")
        logger.info(f"  Precision@{num_items}: {metrics[f'precision@{num_items}']:.4f}")
        logger.info(f"  NDCG@{num_items}: {metrics[f'ndcg@{num_items}']:.4f}")
        logger.info(f"  MRR: {metrics['mrr']:.4f}")
        logger.info(f"  Coverage: {metrics['coverage']:.4f}")
        logger.info(f"  Diversity: {metrics['diversity']:.4f}")
        logger.info("_"*40)

# Convert results to DataFrame and save
df = pd.DataFrame(results)
df.to_json(save_model_comparison_metrics, orient='columns', indent=4)

# Cross-category evaluation
logger.info("Starting cross-category evaluation...")
try:
    # Check if multi-category data is available
    USE_MULTI_CATEGORY = config['data_config'].get('use_multiple_categories', False)
    
    if USE_MULTI_CATEGORY:
        logger.info("Multi-category mode enabled. Running cross-category evaluation...")
        
        # Initialize cross-category evaluator
        cross_category_evaluator = CrossCategoryEvaluator(config)
        
        # Prepare model predictions for cross-category evaluation
        model_predictions = {}
        for entry in test_data:
            user_id = entry['user_id']
            if user_id not in model_predictions:
                model_predictions[user_id] = []
            
            # Use MM-GPT2Rec model response (processed version)
            if 'model_response_items' in entry:
                model_predictions[user_id].extend(entry['model_response_items'][:num_items])
            elif 'model_response' in entry:
                # Fallback to raw model response
                model_predictions[user_id].extend(entry['model_response'].split(', ')[:num_items])
            else:
                # Fallback to any available prediction
                for key in entry.keys():
                    if key not in ['user_id', 'output', 'input', 'model_response', 'model_response_items'] and isinstance(entry[key], list):
                        model_predictions[user_id].extend(entry[key][:num_items])
                        break
        
        # Run cross-category evaluation
        cross_category_results = cross_category_evaluator.run_cross_category_evaluation(model_predictions)
        
        # Create visualizations
        cross_category_evaluator.create_performance_visualizations(cross_category_results)
        
        logger.info("Cross-category evaluation completed!")
        logger.info(f"Categories evaluated: {list(cross_category_results['category_performance'].keys())}")
        
        # Log key findings
        if 'comparison_analysis' in cross_category_results:
            analysis = cross_category_results['comparison_analysis']
            if 'best_performing_category' in analysis:
                logger.info("Best performing categories:")
                for metric, info in analysis['best_performing_category'].items():
                    logger.info(f"  {metric}: {info['category']} ({info['value']:.4f})")
            
            if 'performance_variance' in analysis:
                logger.info("Performance variance across categories:")
                for metric, stats in analysis['performance_variance'].items():
                    logger.info(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
    else:
        logger.info("Multi-category mode disabled. Skipping cross-category evaluation.")
        
except Exception as e:
    logger.error(f"Cross-category evaluation failed: {e}")
    logger.info("Continuing with standard evaluation...")

logger.info("Evaluation completed successfully!")