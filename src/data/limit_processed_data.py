#!/usr/bin/env python3
"""
Script to limit already processed data files to a specific number of entries.
Useful when you have already processed 100k+ entries but want to limit for training.
"""

import json
import random
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def limit_json_file(file_path, max_entries, output_path=None, backup=True):
    """
    Limit a JSON file to a specific number of entries using random sampling.
    
    Args:
        file_path (str): Path to the JSON file
        max_entries (int): Maximum number of entries to keep
        output_path (str): Output path (if None, creates backup and overwrites original)
        backup (bool): Whether to create backup of original file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    # Load data
    logger.info(f"Loading data from {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    original_size = len(data)
    logger.info(f"Original data size: {original_size:,} entries")
    
    if original_size <= max_entries:
        logger.info(f"Data already has {original_size:,} entries (≤ {max_entries:,}), no limiting needed")
        return True
    
    # Random sampling
    logger.info(f"Limiting to {max_entries:,} entries using random sampling")
    random.seed(42)  # For reproducibility
    limited_data = random.sample(data, max_entries)
    
    # Handle output path
    if output_path is None:
        if backup:
            # Create backup of original file
            backup_path = file_path.with_suffix('.json.backup')
            logger.info(f"Creating backup at {backup_path}")
            with open(backup_path, 'w') as f:
                json.dump(data, f, indent=2)
            output_path = file_path
        else:
            output_path = file_path
    else:
        output_path = Path(output_path)
    
    # Save limited data
    with open(output_path, 'w') as f:
        json.dump(limited_data, f, indent=2)
    
    logger.info(f"Saved limited data to {output_path}")
    logger.info(f"Reduced from {original_size:,} to {len(limited_data):,} entries")
    
    return True

def limit_all_processed_data(data_dir, max_entries, backup=True):
    """
    Limit all processed data files (train, val, test) to the same number of entries.
    
    Args:
        data_dir (str): Directory containing processed data
        max_entries (int): Maximum number of entries per file
        backup (bool): Whether to create backups of original files
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return False
    
    # Files to limit
    files_to_limit = [
        "train.json",
        "val.json", 
        "test.json"
    ]
    
    success_count = 0
    for filename in files_to_limit:
        file_path = data_dir / filename
        if file_path.exists():
            logger.info(f"\nProcessing {filename}...")
            if limit_json_file(file_path, max_entries, backup=backup):
                success_count += 1
        else:
            logger.warning(f"File not found: {file_path}")
    
    logger.info(f"\nSuccessfully limited {success_count}/{len(files_to_limit)} files")
    return success_count == len(files_to_limit)

def main():
    parser = argparse.ArgumentParser(description="Limit already processed data files")
    parser.add_argument("max_entries", type=int, 
                       help="Maximum number of entries to keep in each file")
    parser.add_argument("--data-dir", default="data/processed",
                       help="Directory containing processed data files")
    parser.add_argument("--file", 
                       help="Limit specific file instead of all files")
    parser.add_argument("--no-backup", action="store_true",
                       help="Don't create backup files (overwrites originals)")
    parser.add_argument("--output-dir",
                       help="Output directory for limited files (preserves originals)")
    
    args = parser.parse_args()
    
    if args.file:
        # Limit specific file
        if args.output_dir:
            output_path = Path(args.output_dir) / Path(args.file).name
            success = limit_json_file(args.file, args.max_entries, output_path, backup=False)
        else:
            success = limit_json_file(args.file, args.max_entries, backup=not args.no_backup)
    else:
        # Limit all processed data files
        if args.output_dir:
            # Create new files in output directory
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            success = limit_all_processed_data(args.data_dir, args.max_entries, backup=False)
            # Move files to output directory
            for filename in ["train.json", "val.json", "test.json"]:
                src = Path(args.data_dir) / filename
                dst = output_dir / filename
                if src.exists():
                    src.rename(dst)
        else:
            success = limit_all_processed_data(args.data_dir, args.max_entries, backup=not args.no_backup)
    
    if success:
        logger.info("\nData limiting completed successfully!")
        logger.info("You can now run training with the limited dataset.")
    else:
        logger.error("Failed to limit data files.")

if __name__ == "__main__":
    main() 