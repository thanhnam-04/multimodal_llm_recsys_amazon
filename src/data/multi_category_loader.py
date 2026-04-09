"""
Multi-category data loader for Amazon Reviews 2023 dataset.
Addresses the single domain limitation identified by reviewers.
Supports multiple categories from https://amazon-reviews-2023.github.io/
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import logging
import requests
import gzip
import os
from urllib.parse import urlparse
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

class MultiCategoryDataLoader:
    """
    Data loader for multiple Amazon categories.
    Addresses the single domain limitation identified by reviewers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config['data_config']
        
        # Use categories from config; fallback to single-category setup.
        self.categories = self.data_config.get('amazon_categories', ['Digital_Music'])
        self.use_multiple_categories = self.data_config.get('use_multiple_categories', False)
        self.category_balance = self.data_config.get('category_balance', True)
        self.cross_category_evaluation = self.data_config.get('cross_category_evaluation', True)
        
        # Amazon Reviews 2023 URLs - Updated with correct format
        self.base_url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
        self.review_url_template = f"{self.base_url}review_categories/{{category}}.jsonl.gz"
        self.meta_url_template = f"{self.base_url}meta_categories/meta_{{category}}.jsonl.gz"
        
        # Data storage
        self.processed_data = {}
        self.category_stats = {}
        
    def download_category_data(self, category: str, force_download: bool = False) -> Tuple[str, str]:
        """
        Download Amazon Reviews 2023 data for a specific category.
        
        Args:
            category: Category name (e.g., 'Digital_Music')
            force_download: Whether to force re-download even if files exist
            
        Returns:
            Tuple of (review_file_path, meta_file_path)
        """
        data_dir = Path("data/raw")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        review_file = data_dir / f"{category}.jsonl.gz"
        meta_file = data_dir / f"meta_{category}.jsonl.gz"
        
        # Check if files already exist
        if not force_download and review_file.exists() and meta_file.exists():
            logger.info(f"Files already exist for {category}, skipping download")
            return str(review_file), str(meta_file)
        
        # Download review data
        review_url = self.review_url_template.format(category=category)
        logger.info(f"Downloading reviews for {category} from {review_url}")
        
        try:
            response = requests.get(review_url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(review_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded reviews for {category}: {review_file}")
            
        except Exception as e:
            logger.error(f"Failed to download reviews for {category}: {e}")
            raise
        
        # Download metadata
        meta_url = self.meta_url_template.format(category=category)
        logger.info(f"Downloading metadata for {category} from {meta_url}")
        
        try:
            response = requests.get(meta_url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(meta_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded metadata for {category}: {meta_file}")
            
        except Exception as e:
            logger.error(f"Failed to download metadata for {category}: {e}")
            # Clean up partial download
            if review_file.exists():
                review_file.unlink()
            raise
        
        return str(review_file), str(meta_file)
    
    def load_category_data(self, category: str) -> Tuple[List[Dict], Dict[str, Dict]]:
        """
        Load review and metadata for a specific category.
        
        Args:
            category: Amazon category name
            
        Returns:
            Tuple of (reviews_list, metadata_dict)
        """
        review_file, meta_file = self.download_category_data(category)
        
        # Load reviews
        reviews = []
        logger.info(f"Loading reviews from {review_file}...")
        
        with gzip.open(review_file, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        review = json.loads(line)
                        reviews.append(review)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse review: {e}")
                        continue
        
        # Load metadata
        metadata = {}
        logger.info(f"Loading metadata from {meta_file}...")
        
        with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        meta = json.loads(line)
                        metadata[meta['parent_asin']] = meta
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse metadata: {e}")
                        continue
        
        logger.info(f"Loaded {len(reviews)} reviews and {len(metadata)} metadata entries for {category}")
        return reviews, metadata
    
    def process_category_data(self, category: str, reviews: List[Dict], 
                            metadata: Dict[str, Dict]) -> List[Dict]:
        """
        Process raw category data into the expected format with optimized image handling.
        
        Args:
            category: Category name
            reviews: List of review dictionaries
            metadata: Dictionary of metadata by parent_asin
            
        Returns:
            List of processed data entries
        """
        processed_data = []
        
        # First pass: collect all image URLs for batch processing
        image_urls = []
        url_to_review = {}
        
        logger.info(f"Collecting image URLs for {category}...")
        for review in reviews:
            parent_asin = review.get('parent_asin')
            if not parent_asin or parent_asin not in metadata:
                continue
            
            meta = metadata[parent_asin]
            image_urls_batch = self._get_downloadable_images(meta.get('images', []))
            
            if image_urls_batch:
                # Take only the first image for faster processing
                main_image_url = image_urls_batch[0]
                image_urls.append(main_image_url)
                url_to_review[main_image_url] = (review, meta)
        
        # Batch download all images for this category
        if image_urls:
            logger.info(f"Batch downloading {len(image_urls)} images for {category}...")
            from .processor import download_image_batch
            cache_dir = Path(self.data_config.get('image_cache_dir', 'data/processed/image_cache'))
            max_workers = self.data_config.get('batch_download_workers', 20)
            
            # Use optimized batch downloading
            download_results = download_image_batch(image_urls, cache_dir, max_workers)
            logger.info(f"Successfully downloaded {sum(1 for v in download_results.values() if v is not None)}/{len(image_urls)} images for {category}")
        else:
            download_results = {}
            logger.warning(f"No valid image URLs found for {category}")
        
        # Second pass: create processed entries with downloaded image paths
        logger.info(f"Creating processed entries for {category}...")
        for review in reviews:
            parent_asin = review.get('parent_asin')
            if not parent_asin or parent_asin not in metadata:
                continue
            
            meta = metadata[parent_asin]
            
            # Create processed entry with training-compatible format
            title = meta.get('title', '')
            review_text = review.get('text', '')
            
            # Get image URLs
            image_urls_batch = self._get_downloadable_images(meta.get('images', []))
            
            # Only include entries with images
            if not image_urls_batch:
                continue
            
            main_image_url = image_urls_batch[0]
            local_image_path = download_results.get(main_image_url)
            
            entry = {
                'user_id': review.get('user_id'),
                'parent_asin': parent_asin,
                'rating': review.get('rating', 0),
                'title': title,
                'review_text': review_text,
                'category': category,
                'main_category': meta.get('main_category', category),
                'price': meta.get('price', 0),
                'image_urls': [main_image_url],  # Only one image for speed
                'local_image_path': local_image_path,
                'timestamp': review.get('timestamp', 0),
                'verified_purchase': review.get('verified_purchase', False),
                'helpful_vote': review.get('helpful_vote', 0)
            }
            
            processed_data.append(entry)
        
        logger.info(f"Processed {len(processed_data)} entries for {category}")
        return processed_data
    
    def _get_downloadable_images(self, images: List[Dict]) -> List[str]:
        """
        Get downloadable medium resolution images from image metadata.
        
        Args:
            images: List of image metadata dictionaries
            
        Returns:
            List of downloadable image URLs
        """
        downloadable_images = []
        
        for img in images:
            # Try medium resolution first, then fall back to high resolution
            image_url = img.get('med_res') or img.get('hi_res') or img.get('low_res')
            
            if image_url and self._is_downloadable_url(image_url):
                downloadable_images.append(image_url)
                # Only take the first valid image to avoid too many images
                break
        
        return downloadable_images
    
    def _is_downloadable_url(self, url: str) -> bool:
        """
        Check if a URL is likely downloadable.
        
        Args:
            url: Image URL to check
            
        Returns:
            True if URL appears downloadable
        """
        if not url or not isinstance(url, str):
            return False
        
        # Check for common image extensions
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        url_lower = url.lower()
        
        # Must have a valid extension
        if not any(ext in url_lower for ext in valid_extensions):
            return False
        
        # Must be from a trusted domain
        trusted_domains = ['amazon.com', 'media-amazon.com', 'images-amazon.com']
        if not any(domain in url_lower for domain in trusted_domains):
            return False
        
        return True
    
    def process_data_for_training(self, all_data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        Process data using the same logic as original prepare_data.py.
        This creates the proper instruction, input, and output format.
        
        Args:
            all_data: Dictionary of category data
            
        Returns:
            Processed data with proper training format
        """
        logger.info("Processing data for training using original format...")
        
        # Import processor functions
        from .processor import get_next_items, cast_unix_to_date
        
        processed_data = {}
        
        for category, data in all_data.items():
            logger.info(f"Processing {category} data for training format...")
            
            if not data:
                processed_data[category] = []
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Apply the same transformations as original prepare_data.py
            df["date"] = cast_unix_to_date(df["timestamp"])
            df = df.sort_values(by=["date", "user_id"])
            
            # Format parent_asin with ASIN tags
            df["parent_asin"] = "<|ASIN_" + df["parent_asin"] + "|>"
            
            # Create input format: "Review_Text: [review] <|endoftext|>"
            df["input"] = df["review_text"].apply(lambda x: f"Review_Text: {x} <|endoftext|>" if x else "Review_Text: <|endoftext|>")
            
            # Get next items using the same logic as original
            labels_df = get_next_items(
                df=df, 
                padding_strategy='pad_token', 
                pad_token="<|endoftext|>", 
                number_of_items_to_predict=self.config['data_config']['number_of_items_to_predict'], 
                min_interactions=self.config['data_config']['min_interactions']
            )
            
            # Merge with labels
            join_keys = ["user_id", "parent_asin", "date"]
            df = df.merge(labels_df, on=join_keys, how="inner")
            
            # Rename columns to match expected format
            df = df.rename(columns={'next_items': "output", 'next_item_names': "output_names"})
            
            # Add instruction from config
            df['instruction'] = self.config['data_config']['user_prompt']
            
            # Convert back to list of dictionaries
            processed_data[category] = df.to_dict('records')
            
            logger.info(f"Processed {len(processed_data[category])} entries for {category}")
        
        return processed_data
    
    def load_multiple_categories(self) -> Dict[str, List[Dict]]:
        """
        Load data from multiple categories.
        
        Returns:
            Dictionary mapping category names to processed data lists
        """
        all_data = {}
        
        for category in self.categories:
            logger.info(f"Processing category: {category}")
            
            try:
                reviews, metadata = self.load_category_data(category)
                processed_data = self.process_category_data(category, reviews, metadata)
                all_data[category] = processed_data
                
                # Store category statistics
                self.category_stats[category] = {
                    'num_reviews': len(reviews),
                    'num_items': len(metadata),
                    'num_users': len(set(r['user_id'] for r in reviews)),
                    'processed_entries': len(processed_data)
                }
                
            except Exception as e:
                logger.error(f"Failed to process category {category}: {e}")
                continue
        
        return all_data
    
    def balance_categories(self, all_data: Dict[str, List[Dict]], 
                          target_size: int) -> Dict[str, List[Dict]]:
        """
        Balance data across categories to ensure fair representation.
        
        Args:
            all_data: Dictionary of category data
            target_size: Target total size (read from config)
            
        Returns:
            Balanced data dictionary
        """
        if not self.category_balance:
            return all_data
        
        logger.info(f"Balancing categories to target size: {target_size}")
        
        balanced_data = {}
        category_sizes = {}
        
        # Calculate target size per category
        num_categories = len(all_data)
        target_per_category = target_size // num_categories
        
        for category, data in all_data.items():
            if len(data) >= target_per_category:
                # Sample data
                sampled_data = np.random.choice(data, target_per_category, replace=False).tolist()
                balanced_data[category] = sampled_data
                category_sizes[category] = len(sampled_data)
            else:
                # Use all available data
                balanced_data[category] = data
                category_sizes[category] = len(data)
        
        logger.info(f"Balanced category sizes: {category_sizes}")
        return balanced_data
    
    def create_cross_category_splits(self, all_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Create train/val/test splits with cross-category evaluation.
        
        Args:
            all_data: Dictionary of category data
            
        Returns:
            Dictionary containing splits and evaluation setup
        """
        logger.info("Creating cross-category evaluation splits...")
        
        # Combine all data
        combined_data = []
        for category, data in all_data.items():
            for entry in data:
                entry['source_category'] = category
            combined_data.extend(data)
        
        # Shuffle data
        np.random.shuffle(combined_data)
        
        # Create splits
        total_size = len(combined_data)
        train_size = int(total_size * 0.7)
        val_size = int(total_size * 0.15)
        
        train_data = combined_data[:train_size]
        val_data = combined_data[train_size:train_size + val_size]
        test_data = combined_data[train_size + val_size:]
        
        # Create category-specific test sets for cross-category evaluation
        category_test_sets = {}
        for category in self.categories:
            category_test_data = [entry for entry in test_data 
                               if entry.get('source_category') == category]
            category_test_sets[category] = category_test_data
        
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'category_test_sets': category_test_sets,
            'stats': {
                'total_size': total_size,
                'train_size': len(train_data),
                'val_size': len(val_data),
                'test_size': len(test_data),
                'category_sizes': {cat: len(data) for cat, data in category_test_sets.items()}
            }
        }
        
        logger.info(f"Created splits: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        return splits
    
    def save_processed_data(self, splits: Dict[str, Any], output_dir: str = "data/processed"):
        """
        Save processed data splits to files.
        
        Args:
            splits: Data splits dictionary
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main splits
        with open(output_path / 'train.json', 'w') as f:
            json.dump(splits['train'], f, indent=2)
        
        with open(output_path / 'val.json', 'w') as f:
            json.dump(splits['val'], f, indent=2)
        
        with open(output_path / 'test.json', 'w') as f:
            json.dump(splits['test'], f, indent=2)
        
        # Save category-specific test sets
        category_dir = output_path / 'category_tests'
        category_dir.mkdir(exist_ok=True)
        
        for category, test_data in splits['category_test_sets'].items():
            with open(category_dir / f'{category}_test.json', 'w') as f:
                json.dump(test_data, f, indent=2)
        
        # Save statistics
        with open(output_path / 'data_stats.json', 'w') as f:
            json.dump({
                'category_stats': self.category_stats,
                'split_stats': splits['stats'],
                'config': {
                    'categories': self.categories,
                    'use_multiple_categories': self.use_multiple_categories,
                    'category_balance': self.category_balance,
                    'cross_category_evaluation': self.cross_category_evaluation
                }
            }, f, indent=2)
        
        logger.info(f"Saved processed data to {output_path}")
    
    def run_complete_processing(self) -> Dict[str, Any]:
        """
        Run complete data processing pipeline.
        
        Returns:
            Dictionary containing all processed data and statistics
        """
        logger.info("Starting complete multi-category data processing...")
        
        # Load data from multiple categories
        all_data = self.load_multiple_categories()
        
        if not all_data:
            raise ValueError("No data loaded from any category")
        
        # Balance categories if requested
        if self.category_balance:
            target_size = self.config['model_config'].get('test_size', 100000)
            all_data = self.balance_categories(all_data, target_size)
        
        # Process data using the same logic as original prepare_data.py
        processed_data = self.process_data_for_training(all_data)
        
        # Create splits
        splits = self.create_cross_category_splits(processed_data)
        
        # Save processed data
        self.save_processed_data(splits)
        
        # Create special_user_item_ids.json file (required by training script)
        self.create_special_user_item_ids(processed_data)
        
        logger.info("Multi-category data processing completed successfully!")
        
        return {
            'splits': splits,
            'category_stats': self.category_stats,
            'config': self.config
        }
    
    def create_special_user_item_ids(self, processed_data: List[Dict[str, Any]]) -> None:
        """
        Create special_user_item_ids.json file required by the training script.
        This replicates the logic from prepare_data.py.
        """
        logger.info("Creating special_user_item_ids.json file...")
        
        # Extract all unique ASINs from the processed data
        all_asins = set()
        for entry in processed_data:
            if 'asin' in entry:
                all_asins.add(entry['asin'])
            # Also check for parent_asin if it exists
            if 'parent_asin' in entry:
                all_asins.add(entry['parent_asin'])
        
        # Create special user item IDs (same format as prepare_data.py)
        special_user_item_ids = ["<|endoftext|>"] + list(all_asins)
        special_user_item_ids = set(special_user_item_ids)
        
        # Save to the same location as prepare_data.py
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        special_user_item_ids_path = processed_dir / "special_user_item_ids.json"
        
        with open(special_user_item_ids_path, "w") as f:
            json.dump(list(special_user_item_ids), f)
        
        logger.info(f"Saved special_user_item_ids to {special_user_item_ids_path}")
        logger.info(f"Total special tokens: {len(special_user_item_ids)}")


def download_amazon_categories(categories: List[str], output_dir: str = "data/raw") -> Dict[str, bool]:
    """
    Download Amazon Reviews 2023 data for specified categories.
    
    Args:
        categories: List of category names to download
        output_dir: Output directory for downloaded files
        
    Returns:
        Dictionary mapping category names to download success status
    """
    results = {}
    
    for category in categories:
        try:
            loader = MultiCategoryDataLoader({'data_config': {'amazon_categories': [category]}})
            review_file, meta_file = loader.download_category_data(category)
            results[category] = True
            logger.info(f"Successfully downloaded {category}")
        except Exception as e:
            logger.error(f"Failed to download {category}: {e}")
            results[category] = False
    
    return results


if __name__ == "__main__":
    # Example usage
    from src.utils.utils import load_config
    
    config = load_config()
    
    # Create multi-category loader
    loader = MultiCategoryDataLoader(config)
    
    # Run complete processing
    results = loader.run_complete_processing()
    
    print("Multi-category data processing completed!")
    print(f"Categories processed: {list(results['category_stats'].keys())}")
    print(f"Total data size: {results['splits']['stats']['total_size']}")
    
    # Example: Download specific categories
    categories_to_download = ['Digital_Music']
    download_results = download_amazon_categories(categories_to_download)
    print(f"Download results: {download_results}")
