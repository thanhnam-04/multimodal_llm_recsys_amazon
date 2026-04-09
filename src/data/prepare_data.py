from tqdm import tqdm
import logging
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tiktoken
import os

from ..utils.utils import setup_logging, set_seed, load_config
from . import processor

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)
set_seed(42)

# Test the logger
logger.info("Starting data preparation script")

## TODO: 
# - Add a function to get the next 10 items for each user
# - USE Multi-head attention with PyTorch's scaled dot product attention and FlashAttention


def _resolve_raw_category_files(raw_dir: Path, config: dict):
    """Resolve review/meta files for the first available configured category.

    Supports both .jsonl and .jsonl.gz files. If local files are missing,
    attempts to download the first configured category via MultiCategoryDataLoader.
    """
    categories = config.get('data_config', {}).get('amazon_categories', ['Digital_Music'])

    for category in categories:
        for ext in ('.jsonl', '.jsonl.gz'):
            review_file = raw_dir / f"{category}{ext}"
            meta_file = raw_dir / f"meta_{category}{ext}"
            if review_file.exists() and meta_file.exists():
                return category, review_file, meta_file

    try:
        from .multi_category_loader import MultiCategoryDataLoader

        target_category = categories[0]
        logger.info(
            f"No local raw files found. Downloading category '{target_category}'..."
        )
        loader = MultiCategoryDataLoader(config)
        review_file, meta_file = loader.download_category_data(target_category)
        return target_category, Path(review_file), Path(meta_file)
    except Exception as exc:
        expected = [
            f"{category}.jsonl(.gz) + meta_{category}.jsonl(.gz)"
            for category in categories
        ]
        raise FileNotFoundError(
            "No valid raw files were found in data/raw and auto-download failed. "
            f"Expected one of: {', '.join(expected)}. "
            f"Download error: {exc}"
        ) from exc


def _has_valid_local_image(path_value) -> bool:
    if not path_value:
        return False
    path_str = str(path_value).strip()
    if not path_str:
        return False
    return Path(path_str).exists()


def _filter_records_with_images(records):
    return [row for row in records if _has_valid_local_image(row.get("local_image_path"))]



def main():

    # STEP 0: Load data
    # ==========================================
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    cache_dir = processed_dir / "image_cache"


    # Create necessary directories
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()

    # Test mode parameters
    TEST_MODE = config['model_config']['test_mode']
    #TEST_MODE = TEST_MODE.lower()
    #TEST_MODE = json.loads(TEST_MODE)
    TEST_SIZE = int(config['model_config']['test_size'])   # Number of items to process in test mode

    # Check if multi-category mode is enabled
    USE_MULTI_CATEGORY = config['data_config'].get('use_multiple_categories', False)
    REQUIRE_IMAGES_ONLY = config['data_config'].get('require_images_only', False)
    
    # Check if data already exists (check for both old and new formats)
    train_file = processed_dir / "train.json"
    val_file = processed_dir / "val.json"
    test_file = processed_dir / "test.json"
    processed_data_file = processed_dir / "processed_data.jsonl"
    special_ids_file = processed_dir / "special_user_item_ids.json"
    
    # Check if we have the split files OR the processed data file
    data_exists = (train_file.exists() and val_file.exists() and test_file.exists()) or processed_data_file.exists()
    
    if data_exists:
        logger.info("Data files already exist, loading from cache...")
        
        # Check which format we have
        if train_file.exists() and val_file.exists() and test_file.exists():
            logger.info(f"   - Train: {train_file}")
            logger.info(f"   - Val: {val_file}")
            logger.info(f"   - Test: {test_file}")
            
            # Load existing split data
            with open(train_file, 'r') as f:
                train_data = json.load(f)
            with open(val_file, 'r') as f:
                val_data = json.load(f)
            with open(test_file, 'r') as f:
                test_data = json.load(f)

            if REQUIRE_IMAGES_ONLY:
                train_before, val_before, test_before = len(train_data), len(val_data), len(test_data)
                train_data = _filter_records_with_images(train_data)
                val_data = _filter_records_with_images(val_data)
                test_data = _filter_records_with_images(test_data)

                logger.info(
                    "Image-only filtering applied to cached splits: "
                    f"Train {train_before}->{len(train_data)}, "
                    f"Val {val_before}->{len(val_data)}, "
                    f"Test {test_before}->{len(test_data)}"
                )

                with open(train_file, 'w') as f:
                    json.dump(train_data, f, indent=2)
                with open(val_file, 'w') as f:
                    json.dump(val_data, f, indent=2)
                with open(test_file, 'w') as f:
                    json.dump(test_data, f, indent=2)
            
            review_data = train_data + val_data + test_data
            logger.info(f"Loaded {len(review_data)} total records from split files")
            
        elif processed_data_file.exists():
            logger.info(f"   - Processed data: {processed_data_file}")
            
            # Load existing processed data
            review_data = []
            with open(processed_data_file, 'r') as f:
                for line in f:
                    review_data.append(json.loads(line.strip()))

            if REQUIRE_IMAGES_ONLY:
                before_count = len(review_data)
                review_data = _filter_records_with_images(review_data)
                logger.info(
                    f"Image-only filtering applied to cached processed data: {before_count}->{len(review_data)}"
                )
            
            logger.info(f"Loaded {len(review_data)} total records from processed data")
            
            # Generate train/test/val splits from processed data
            logger.info("Generating train/test/val splits from processed data...")
            train_data, val_data, test_data = split_data(review_data, config)
            
            # Save the splits
            with open(train_file, 'w') as f:
                json.dump(train_data, f, indent=2)
            with open(val_file, 'w') as f:
                json.dump(val_data, f, indent=2)
            with open(test_file, 'w') as f:
                json.dump(test_data, f, indent=2)
            
            logger.info(f"Generated splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
            
            # Generate statistics
            logger.info("Generating data statistics...")
            stats = generate_data_statistics(train_data, val_data, test_data, review_data)
            stats_file = processed_dir / "data_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Statistics saved to: {stats_file}")
        
        # Only create special_user_item_ids.json if missing
        if not special_ids_file.exists():
            logger.info("Creating missing special_user_item_ids.json...")
            special_user_item_ids = ["<|endoftext|>"] + list(set([entry.get('asin', '') for entry in review_data if entry.get('asin')]))
            with open(special_ids_file, 'w') as f:
                json.dump(special_user_item_ids, f, indent=2)
            logger.info(f"Created special_user_item_ids.json with {len(special_user_item_ids)} items")
        
        logger.info("Data preparation completed using cached data!")
        return

    # Load data
    category_name, review_file, meta_file = _resolve_raw_category_files(raw_dir, config)
    logger.info(f"Using category: {category_name}")
    logger.info(f"Loading review data from {review_file}")
    review_data = processor.load_jsonl(review_file)


    if TEST_MODE:
        logger.info(f"Running in test mode with {TEST_SIZE} items")
        # Take a random sample of reviews
        review_data = review_data[:TEST_SIZE]

    # Filter users with at least min_interactions interactions
    logger.info(f"Filtering users with at least {config['data_config']['min_interactions']} interactions...")
    filtered_review_data = processor.filter_users_by_interactions(review_data, min_interactions=config['data_config']['min_interactions'])
    
    # Additional filtering for longer sequences if enabled
    if config['data_config'].get('sequence_aware_training', False):
        min_items_per_sequence = config['data_config'].get('min_items_per_sequence', 8)
        logger.info(f"Additional filtering for users with {min_items_per_sequence}+ items per sequence...")
        filtered_review_data = processor.filter_users_by_sequence_length(filtered_review_data, min_items_per_sequence=min_items_per_sequence)

    # Get unique parent_asins from filtered reviews
    valid_parent_asins = set(review.get('parent_asin') for review in filtered_review_data)
    
    logger.info(f"Loading metadata from {meta_file}")
    metadata = processor.load_jsonl(meta_file)


    if TEST_MODE:
        # Filter metadata to only include items from test reviews
        metadata = [item for item in metadata 
                   if item.get('parent_asin') in valid_parent_asins]
        metadata = metadata[:TEST_SIZE]


    # STEP 1: Filtering and processing data
    # ==========================================

    # > Filter metadata to only include items from filtered reviews
    filtered_metadata = [item for item in metadata 
                        if item.get('parent_asin') in valid_parent_asins]
    logger.info(f"Filtered metadata items: {len(filtered_metadata)}")

    # 1.01 Process reviews and metadata
    # Process data
    logger.info("Processing review data...")
    review_df = processor.process_review_data(filtered_review_data)
    
    logger.info("Processing metadata and caching images...")
    metadata_df = processor.process_metadata(filtered_metadata, cache_dir)
    
    # Final merge data: review and metadata
    merged_df = processor.merge_data(review_df, metadata_df)

    logger.info(f"Final merged dataframe shape: {merged_df.shape}")

    if REQUIRE_IMAGES_ONLY:
        before_rows = len(merged_df)
        merged_df = merged_df[
            merged_df["local_image_path"].apply(_has_valid_local_image)
        ].copy()
        logger.info(f"Image-only filtering applied: {before_rows}->{len(merged_df)} rows")
        if merged_df.empty:
            raise ValueError(
                "No rows with valid local images after filtering. "
                "Try re-running data preparation to refresh image cache."
            )

    
    # 1.02 Data user and item to custom format
    # Prepare the data by setting the user_id and item_id to a special token format
    # ==========================================

    #cols_to_drop = ["timestamp", "parent_asin", "user_id", "details", "local_image_path"]
    
    review_cols = ["review_text"]
    item_cols = []
    
    merged_df["date"] = processor.cast_unix_to_date(merged_df["timestamp"])

    merged_df = merged_df.sort_values(by=["date", "user_id"])
    #merged_df['parent_asin_original'] = merged_df['parent_asin'].copy()
    merged_df["parent_asin"] = "<|ASIN_" + merged_df["parent_asin"] + "|>"
    #merged_df['input'] = merged_df[review_cols + item_cols].apply(details_to_sentence)
    merged_df["input"] = merged_df[review_cols + item_cols].apply(lambda x: '\n'.join(f"{col.title()}: {value} <|endoftext|>" for col, value in x.items()), axis=1)

   

    # > Prepare the data labels: Extract the next items purchased by the user
    labels_df = processor.get_next_items(
        df=merged_df, padding_strategy='pad_token', pad_token="<|endoftext|>", 
        number_of_items_to_predict=config['data_config']['number_of_items_to_predict'], 
        min_interactions=config['data_config']['min_interactions'])
    
    join_keys = ["user_id", "parent_asin", "date"]
    merged_df = merged_df.merge(labels_df, on=join_keys, how="inner")

    #_________
    # Save parent_asin and product title to JSON
    parent_asin_title_path = processed_dir / "parent_asin_title.json"
    parent_asin_title_data = merged_df[['parent_asin', 'title']].to_dict(orient='records')
    with open(parent_asin_title_path, 'w') as f:
        json.dump(parent_asin_title_data, f)
    logger.info(f"Saved parent_asin and product title to {parent_asin_title_path}")

    #__________

    merged_df = merged_df.drop(columns=review_cols + item_cols)
    merged_df = merged_df.drop(columns=["timestamp"])

    del labels_df

    logger.info(f"Finished processing data and collecting labels: {merged_df.shape}")

    # 1.03 Tokenizer
    #        Lets the identify user_id, item and "<|endoftext|>" as additional tokens
    #_______________________________________________________________________________
    special_user_item_ids = ["<|endoftext|>"] + merged_df["parent_asin"].unique().tolist()
    special_user_item_ids = set(special_user_item_ids)

    #tokenizer = tiktoken.get_encoding("gpt2")   

    # Add special tokens to the tokenizer
    #end_of_text_id = np.array(tokenizer.encode("<|endoftext|>", allowed_special=special_user_item_ids)).item()


    # 2. DATA PREP - Data Loaders, Custom collate, etc.
    # ==========================================

    ## 2.01 Data preparation - input, output and instruction
    #_______________________________________________________________________________

    df_model_ready = merged_df.copy()

    # Create proper input format matching LLMs-for-RecSys approach
    # Include item details in the input for better context
    def create_input_text(row):
        parts = [f"<|user_{row['user_id']}|>"]
        
        # Add item details
        if pd.notna(row.get('parent_asin')):
            parts.append(f"Item id is {row['parent_asin']}.")
        if pd.notna(row.get('rating')):
            parts.append(f"Rating of the item by user from 1 to 5 is {row['rating']}.")
        if pd.notna(row.get('review_title')):
            parts.append(f"Review title: {row['review_title']}.")
        if pd.notna(row.get('title')):
            parts.append(f"Item name is {row['title']}.")
        if pd.notna(row.get('main_category')):
            parts.append(f"Main category of the item is {row['main_category']}.")
        
        return " ".join(parts)
    
    df_model_ready['input'] = df_model_ready.apply(create_input_text, axis=1)
    
    # Rename columns to match expected format
    df_model_ready = df_model_ready.rename(columns={'next_items': "output", 'next_item_names': "output_names"})
    df_model_ready['instruction'] = config['data_config']['user_prompt']


    # Save processed data with test mode indicator
    output_filename = "processed_data_test.jsonl" if TEST_MODE else "processed_data.jsonl"
    output_path = processed_dir / output_filename
    processor.save_processed_data(df_model_ready, output_path)
    logger.info(f"Saved processed data to {output_path}")

    # 2.02 Split to train, validation and test
    #_______________________________________________________________________________
    train_df, val_df, test_df = processor.split_data_temporal(
        df_model_ready, 
        train_ratio=config['data_split_ratio']['train_ratio'], 
        val_ratio=config['data_split_ratio']['val_ratio'], 
        test_ratio=config['data_split_ratio']['test_ratio'])

    train_data = train_df.to_dict(orient="records")
    test_data = test_df.to_dict(orient="records")
    val_data = val_df.to_dict(orient="records")

    del train_df, val_df, test_df

    processor.save_split_data(train_data, val_data, test_data, str(processed_dir))

    special_user_item_ids_path = processed_dir / "special_user_item_ids.json"    
    
    with open(special_user_item_ids_path, "w") as f:
        json.dump(list(special_user_item_ids), f)
    logger.info(f"Saved special_user_item_ids to {special_user_item_ids_path}")



    # Log split statistics
    logger.info(f"Training set size: {len(train_data)}")
    logger.info(f"Validation set size: {len(val_data)}")
    logger.info(f"Test set size: {len(test_data)}")
    
    logger.info("Data preparation and splitting completed successfully!")







def split_data(data, config):
    """Split data into train/val/test sets."""
    from sklearn.model_selection import train_test_split
    
    # Get split ratios
    train_ratio = config['data_split_ratio']['train_ratio']
    val_ratio = config['data_split_ratio']['val_ratio']
    test_ratio = config['data_split_ratio']['test_ratio']
    
    # First split: train vs (val + test)
    train_data, temp_data = train_test_split(
        data, 
        test_size=(val_ratio + test_ratio),
        random_state=42
    )
    
    # Second split: val vs test
    val_data, test_data = train_test_split(
        temp_data,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=42
    )
    
    return train_data, val_data, test_data


def generate_data_statistics(train_data, val_data, test_data, all_data):
    """Generate comprehensive data statistics."""
    stats = {
        "dataset_overview": {
            "total_records": len(all_data),
            "train_records": len(train_data),
            "val_records": len(val_data),
            "test_records": len(test_data),
            "train_ratio": len(train_data) / len(all_data),
            "val_ratio": len(val_data) / len(all_data),
            "test_ratio": len(test_data) / len(all_data)
        },
        "user_statistics": {},
        "item_statistics": {},
        "category_statistics": {},
        "text_statistics": {}
    }
    
    # User statistics
    all_users = set()
    all_items = set()
    all_categories = set()
    text_lengths = []
    
    for entry in all_data:
        if 'user_id' in entry:
            all_users.add(entry['user_id'])
        if 'asin' in entry:
            all_items.add(entry['asin'])
        if 'category' in entry:
            all_categories.add(entry['category'])
        if 'input' in entry:
            text_lengths.append(len(entry['input']))
    
    stats["user_statistics"] = {
        "unique_users": len(all_users),
        "avg_interactions_per_user": len(all_data) / len(all_users) if all_users else 0
    }
    
    stats["item_statistics"] = {
        "unique_items": len(all_items),
        "avg_interactions_per_item": len(all_data) / len(all_items) if all_items else 0
    }
    
    stats["category_statistics"] = {
        "unique_categories": len(all_categories),
        "categories": list(all_categories)
    }
    
    if text_lengths:
        stats["text_statistics"] = {
            "avg_text_length": sum(text_lengths) / len(text_lengths),
            "min_text_length": min(text_lengths),
            "max_text_length": max(text_lengths),
            "median_text_length": sorted(text_lengths)[len(text_lengths)//2]
        }
    
    return stats


if __name__ == "__main__":
    main()
