# MM-GPT2Rec Source Code Documentation

This document provides a comprehensive guide to the MM-GPT2Rec source code structure, functionality, and usage instructions.

## Project Structure

```
src/
├── __init__.py                 # Package initialization
├── baselines/                   # Baseline model implementations
├── data/                       # Data processing and preparation
├── evaluation/                 # Model evaluation and metrics
├── models/                     # Core model architecture
├── training/                   # Training scripts and utilities
└── utils/                      # Utility functions and helpers
```

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (recommended)
- Required packages: `transformers`, `torch`, `pandas`, `numpy`, `tqdm`, `PIL`

### Installation
```bash
pip install -r requirements.txt
```

### Complete Workflow
```bash
# Run the complete pipeline
python run_enhanced_workflow.py --steps all
```

## Detailed Module Documentation

### `baselines/` - Baseline Model Implementations

Contains implementations of various baseline recommendation models for comparison.

#### Files:
- **`multimodal_baselines.py`**: Full neural network implementations of multimodal baselines
- **`fast_baselines.py`**: Optimized, vectorized implementations for faster evaluation
- **`recommender.py`**: Traditional recommendation algorithms (CF, Content-Based, etc.)
- **`run_baselines.py`**: Script to run baseline evaluations

#### Key Classes:
- `VBPR`: Visual Bayesian Personalized Ranking
- `DeepCoNN`: Deep Cooperative Neural Networks
- `NRMF`: Neural Recommendation with Matrix Factorization
- `SASRec`: Self-Attentive Sequential Recommendation
- `FastVBPR`, `FastDeepCoNN`, `FastNRMF`, `FastSASRec`: Optimized versions

#### Usage:
```python
from src.baselines.multimodal_baselines import create_multimodal_baselines
from src.baselines.fast_baselines import create_fast_baselines

# Create baselines
baselines, user_mapping, item_mapping = create_fast_baselines(config)
```

#### Expected Outputs:
- Baseline model predictions in `results/test_with_baseline_predictions.json`
- Performance metrics for each baseline model

---

### `data/` - Data Processing and Preparation

Handles all data loading, preprocessing, and preparation for training and evaluation.

#### Files:
- **`prepare_data.py`**: Main data preparation script
- **`processor.py`**: Core data processing logic
- **`dataset.py`**: PyTorch dataset classes
- **`multi_category_loader.py`**: Multi-category data loading utilities
- **`limit_processed_data.py`**: Data limiting utilities

#### Key Functions:
- `prepare_data()`: Downloads and processes Amazon product data
- `get_next_items()`: Generates target sequences for training
- `create_input_text()`: Formats input sequences with item details
- `split_data_temporal()`: Per-user temporal data splitting

#### Usage:
```bash
# Run data preparation
python -m src.data.prepare_data

# Or use the main workflow
python run_enhanced_workflow.py --steps data
```

#### Expected Outputs:
- `data/processed/train.json`: Training data
- `data/processed/val.json`: Validation data  
- `data/processed/test.json`: Test data
- `data/processed/parent_asin_title.json`: Product catalog
- `data/processed/image_cache/`: Cached image features

#### Configuration:
Key parameters in `configs/train_config.json`:
- `min_interactions`: Minimum user interactions (default: 5)
- `number_of_items_to_predict`: Prediction sequence length (default: 5)
- `categories`: Product categories to include

---

### `evaluation/` - Model Evaluation and Metrics

Comprehensive evaluation framework with multiple metrics and analysis tools.

#### Files:
- **`basic_evaluation.py`**: Standard recommendation metrics
- **`compare_baselines.py`**: Baseline comparison evaluation
- **`ablation_studies.py`**: Ablation study analysis
- **`cross_category_evaluation.py`**: Category-specific evaluation
- **`main_evaluation.py`**: Main evaluation orchestrator
- **`metrics.py`**: Two-track evaluation metrics (legacy)
- **`metrics_v2.py`**: Updated metrics implementation
- **`process_model_outputs.py`**: Model output processing
- **`robust_evaluation.py`**: Statistical evaluation with confidence intervals

#### Key Metrics:
- **Accuracy Metrics**: Hit Rate@K, Precision@K, Recall@K, NDCG@K, MRR, MAP@K
- **Beyond-Accuracy**: Coverage, Diversity, Novelty
- **Statistical**: Confidence intervals, significance testing

#### Usage:
```bash
# Run basic evaluation
python -m src.evaluation.basic_evaluation

# Run baseline comparison
python -m src.evaluation.compare_baselines

# Run ablation studies
python -m src.evaluation.ablation_studies

# Run cross-category evaluation
python -m src.evaluation.cross_category_evaluation
```

#### Expected Outputs:
- `results/basic_evaluation_results.json`: Basic metrics
- `results/test_with_baseline_predictions.json`: Baseline predictions
- `results/ablation_study_results.json`: Ablation study results
- `results/cross_category_evaluation_results.json`: Category analysis
- `results/result_metrics.json`: Comprehensive results summary

---

### `models/` - Core Model Architecture

Contains the main MM-GPT2Rec model implementation.

#### Files:
- **`model.py`**: GPT-2 based multimodal recommendation model

#### Key Classes:
- `MultimodalGPT2`: Main model class extending GPT-2
- `MultimodalEmbedding`: Item embedding with text and image fusion
- `LoRAAdapter`: Parameter-efficient fine-tuning adapter

#### Usage:
```python
from src.models.model import MultimodalGPT2

# Initialize model
model = MultimodalGPT2(config)
```

#### Model Architecture:
- **Base**: GPT-2 Medium (355M parameters)
- **Multimodal Fusion**: Concatenation + linear layer
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Input**: Item sequences with multimodal embeddings
- **Output**: Next-item probability distributions

---

### `training/` - Training Scripts and Utilities

Handles model training, fine-tuning, and output processing.

#### Files:
- **`train.py`**: Main training script
- **`process_outputs.py`**: Training output processing utilities

#### Key Features:
- **LoRA Fine-tuning**: Parameter-efficient training
- **Mixed Precision**: FP16 training for memory efficiency
- **Gradient Checkpointing**: Memory optimization
- **Image Caching**: Efficient image feature loading
- **Multi-GPU Support**: Distributed training capabilities

#### Usage:
```bash
# Run training
python -m src.training.train

# Or use the main workflow
python run_enhanced_workflow.py --steps train
```

#### Expected Outputs:
- `checkpoints/`: Model checkpoints
- `logs/training.log`: Training logs
- `data/processed/test_with_responses.json`: Model predictions
- `data/processed/test_with_responses_processed.json`: Processed predictions

#### Training Configuration:
- **Epochs**: 20
- **Learning Rate**: 1e-5 (with cosine scheduling)
- **Batch Size**: 4 (with gradient accumulation)
- **LoRA Rank**: 8
- **Mixed Precision**: FP16

---

### `utils/` - Utility Functions and Helpers

Common utilities used across the project.

#### Files:
- **`utils.py`**: Core utility functions
- **`visualization.py`**: Plotting and visualization utilities

#### Key Functions:
- `setup_logging()`: Logging configuration
- `load_config()`: Configuration loading
- `set_seed()`: Random seed setting
- `download_image()`: Image downloading utilities
- `create_plots()`: Evaluation result visualization

#### Usage:
```python
from src.utils.utils import setup_logging, load_config, set_seed

# Setup logging
setup_logging()

# Load configuration
config = load_config()

# Set random seed
set_seed(42)
```

---

## Complete Workflow

### 1. Data Preparation
```bash
python -m src.data.prepare_data
```
**Outputs**: Processed datasets, image cache, product catalog

### 2. Model Training
```bash
python -m src.training.train
```
**Outputs**: Trained model checkpoints, training logs

### 3. Model Evaluation
```bash
python -m src.evaluation.basic_evaluation
python -m src.evaluation.compare_baselines
python -m src.evaluation.ablation_studies
```
**Outputs**: Comprehensive evaluation results

### 4. Analysis and Visualization
```bash
python analyze_predictions_with_names.py
```
**Outputs**: Prediction examples with product names

## Configuration

### Main Configuration File: `configs/train_config.json`

```json
{
  "data_config": {
    "min_interactions": 5,
    "number_of_items_to_predict": 5,
    "categories": ["Appliances", "Digital Music", "Gift Cards", "Health and Personal Care"]
  },
  "model_config": {
    "base_model": "gpt2-medium",
    "multimodal_fusion": "concatenation",
    "lora_rank": 8
  },
  "training_config": {
    "num_epochs": 20,
    "learning_rate": 1e-5,
    "batch_size": 4
  }
}
```

## Expected Results

### Performance Metrics (k=5):
- **Hit Rate@5**: 83.3%
- **Precision@5**: 29.7%
- **NDCG@5**: 22.0%
- **MRR**: 20.7%
- **Coverage**: 71.5%
- **Diversity**: 74.9%

### Baseline Comparison:
- **MM-GPT2Rec**: 83.3% HR@5
- **Multimodal Baselines**: ~45.6% HR@5
- **Traditional Baselines**: 1.1-9.3% HR@5

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   - Reduce batch size in config
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Image Download Failures**:
   - Check internet connection
   - Verify image URLs in dataset
   - Use cached images when available

3. **Slow Baseline Evaluation**:
   - Use `fast_baselines.py` instead of `multimodal_baselines.py`
   - Enable GPU acceleration
   - Reduce evaluation sample size

### Log Files:
- `logs/data_prep.log`: Data preparation logs
- `logs/training.log`: Training logs
- `logs/evaluation.log`: Evaluation logs

## Additional Resources

- **Main README**: `README.md` - Project overview and installation
- **Paper**: `paper/IEEE_Multimodaln_RecSys/conference_101719.tex` - Research paper
- **Results**: `results/` - All evaluation results and outputs
- **Configs**: `configs/` - Configuration files

## Contributing

When modifying the codebase:

1. Follow the existing code structure
2. Update this documentation for new features
3. Add appropriate logging
4. Test with the complete workflow
5. Update configuration files as needed

## Support

For questions or issues:
1. Check the logs in `logs/` directory
2. Review configuration in `configs/train_config.json`
3. Verify data integrity in `data/processed/`
4. Check GPU memory usage and system resources
