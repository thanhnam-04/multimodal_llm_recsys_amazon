import pandas as pd
import torch
import math
import os
import json
import gzip
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Union
import requests
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# DATA PROCESSING UTILITIES:
# ==========================================

def cast_unix_to_date(unix_time_str: pd.Series) -> pd.Series:
    """
    Converts a Series of Unix timestamps (milliseconds) to datetime strings.

    Args:
        unix_time_str (pd.Series): Series of Unix timestamps as strings.

    Returns:
        pd.Series: Series of datetime strings in the format 'YYYY-MM-DD'.
    """

    # Convert to numeric directly
    unix_timestamps = pd.to_numeric(unix_time_str)

    # Convert to datetime using NumPy for efficiency
    datetime_values = pd.to_datetime(unix_timestamps / 1000, unit='s')  # Convert milliseconds to seconds

    # Format as strings
    return datetime_values.dt.strftime('%Y-%m-%d')

def filter_users_by_interactions(review_data: List[Dict], min_interactions: int = 3) -> List[Dict]:
    """Filter users to keep only those with at least min_interactions."""
    # Count interactions per user
    user_interactions = defaultdict(int)
    for review in review_data:
        user_id = review.get('user_id')
        if user_id:
            user_interactions[user_id] += 1
    
    # Filter users with enough interactions
    valid_users = {user_id for user_id, count in user_interactions.items() 
                  if count >= min_interactions}
    
    # Filter review data
    filtered_data = [review for review in review_data 
                    if review.get('user_id') in valid_users]
    
    logger.info(f"Original users: {len(user_interactions)}")
    logger.info(f"Users with >= {min_interactions} interactions: {len(valid_users)}")
    logger.info(f"Original reviews: {len(review_data)}")
    logger.info(f"Filtered reviews: {len(filtered_data)}")
    
    return filtered_data

def filter_users_by_sequence_length(review_data: List[Dict], min_items_per_sequence: int = 8) -> List[Dict]:
    """
    Filter users to keep only those with sequences that have at least min_items_per_sequence items.
    This ensures the model sees longer sequences during training.
    """
    # Group by user and create sequences
    user_sequences = defaultdict(list)
    for review in review_data:
        user_id = review.get('user_id')
        if user_id:
            user_sequences[user_id].append(review)
    
    # Filter users with long enough sequences
    valid_users = set()
    for user_id, interactions in user_sequences.items():
        if len(interactions) >= min_items_per_sequence:
            valid_users.add(user_id)
    
    # Filter review data
    filtered_data = [review for review in review_data 
                    if review.get('user_id') in valid_users]
    
    logger.info(f"Original users: {len(user_sequences)}")
    logger.info(f"Users with {min_items_per_sequence}+ items per sequence: {len(valid_users)}")
    logger.info(f"Original reviews: {len(review_data)}")
    logger.info(f"Filtered reviews: {len(filtered_data)}")
    
    return filtered_data

def create_longer_sequences(df: pd.DataFrame, min_sequence_length: int = 8) -> pd.DataFrame:
    """
    Create longer sequences by combining multiple user interactions.
    This helps the model learn to predict more items before stopping.
    """
    df_sorted = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    enhanced_data = []
    
    for user_id, user_data in df_sorted.groupby('user_id'):
        user_data = user_data.reset_index(drop=True)
        
        if len(user_data) < min_sequence_length:
            continue
            
        # Create sliding window sequences
        for i in range(len(user_data) - min_sequence_length + 1):
            sequence = user_data.iloc[i:i + min_sequence_length]
            
            # Create input sequence (first 70% of items)
            input_length = int(len(sequence) * 0.7)
            input_sequence = sequence.iloc[:input_length]
            
            # Create target sequence (remaining 30% of items)
            target_sequence = sequence.iloc[input_length:]
            
            # Add to enhanced data
            enhanced_data.append({
                'user_id': user_id,
                'input_sequence': input_sequence['parent_asin'].tolist(),
                'target_sequence': target_sequence['parent_asin'].tolist(),
                'input_titles': input_sequence['title'].tolist(),
                'target_titles': target_sequence['title'].tolist(),
                'timestamp': sequence.iloc[0]['timestamp']
            })
    
    return pd.DataFrame(enhanced_data)


def get_next_items(
    df: pd.DataFrame,
    padding_strategy='repeat',
    pad_token="<|endoftext|>",
    number_of_items_to_predict=10,
    min_interactions=10
):
    """
    For each row in the dataframe, get the next N items the user interacted with.
    Based on the successful LLMs-for-RecSys approach.
    
    Args:
        df (pd.DataFrame): Must have 'user_id', 'timestamp', 'parent_asin', 'title'
        padding_strategy (str): 'none', 'repeat', 'pad_token'
        pad_token (str): Value to pad with if applicable
        number_of_items_to_predict (int): Number of items to get
        min_interactions (int): Minimum interactions per user
    
    Returns:
        pd.DataFrame with ['user_id', 'parent_asin', 'timestamp', 'next_items', 'next_item_names']
    """
    df_sorted = df.sort_values(by=['timestamp']).reset_index(drop=True)
    next_items_data = []

    for user_id, user_data in df_sorted.groupby('user_id'):
        user_data = user_data.reset_index(drop=True)
        num_interactions = len(user_data)

        if num_interactions < min_interactions:
            continue

        # Process items where we have enough future interactions
        # Allow users with exactly min_interactions to contribute at least one sample
        max_start_idx = max(0, num_interactions - number_of_items_to_predict)
        for i in range(max_start_idx + 1):
            # Get next items starting from i+1
            next_df = user_data.loc[i + 1:i + number_of_items_to_predict, :]
            next_items = next_df['parent_asin'].tolist()
            next_item_names = next_df['title'].tolist()

            # Ensure we have exactly number_of_items_to_predict items
            if len(next_items) < number_of_items_to_predict:
                num_missing = number_of_items_to_predict - len(next_items)
                
                if padding_strategy == 'repeat':
                    if len(next_items) > 0:
                        next_items += [next_items[-1]] * num_missing
                        next_item_names += [next_item_names[-1]] * num_missing
                    else:
                        # Can't repeat from an empty list, fallback to pad_token
                        next_items += [pad_token] * number_of_items_to_predict
                        next_item_names += [pad_token] * number_of_items_to_predict

                elif padding_strategy == 'pad_token':
                    next_items += [pad_token] * num_missing
                    next_item_names += [pad_token] * num_missing

                elif padding_strategy == 'none':
                    # Skip rows where we don't have enough next items
                    continue

            next_items_data.append((
                user_data.loc[i, 'user_id'],
                user_data.loc[i, 'parent_asin'],
                user_data.loc[i, 'date'],
                ", ".join(next_items),
                "| ".join(next_item_names)
            ))

    df_next_items = pd.DataFrame(
        next_items_data,
        columns=['user_id', 'parent_asin', 'date', 'next_items', 'next_item_names']
    )

    return df_next_items


def details_to_sentence(details_dict):
    sentences = [f"{key.title()}: {value}\n" for key, value in details_dict.items()]
    history = "\n\n".join(sentences)
    return history
    

def format_input_alpaca(entry):
    """ 
    Format entry according to the Alpaca-style prompt template. Example of data entry is as follows:

    Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    Identify the correct spelling of the following word.

    ### Input:
    Occassion

    ### Response:
    The correct spelling is 'Occasion.'
    """
    
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def format_input_phi(entry):
    """
    Format entry according to the Phi-3-style prompt template.
    """
    # Handle missing fields gracefully
    instruction = entry.get('instruction', 'Given a user\'s purchase history and review for a product, predict the next 10 products they would likely purchase.')
    input_text = entry.get('input', '')
    
    # Create Phi-3 format (no truncation - handled in training)
    formatted_text = f"<|user|>\n{instruction}\n{input_text}<|end|>\n<|assistant|>\n"
    
    return formatted_text


def split_data_temporal(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits the data into train, validation, and test sets PER USER based on temporal order.
    This ensures proper item overlap between train and test while maintaining temporal ordering.
    
    Matches the approach from successful LLMs-for-RecSys project.

    Args:
    - df (pd.DataFrame): DataFrame containing user interactions with 'user_id' and 'date' columns.
    - train_ratio (float): Proportion of data to be used for training.
    - val_ratio (float): Proportion of data to be used for validation.
    - test_ratio (float): Proportion of data to be used for testing.

    Returns:
    - train_df (pd.DataFrame): Training set.
    - val_df (pd.DataFrame): Validation set.
    - test_df (pd.DataFrame): Test set.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum up to 1."

    # Convert date strings to datetime objects
    df['date'] = pd.to_datetime(df['date'])

    # Split PER USER to ensure item overlap
    train_list = []
    val_list = []
    test_list = []
    
    for user_id, user_data in df.groupby('user_id'):
        # Sort user's data by date
        user_sorted = user_data.sort_values('date').reset_index(drop=True)
        n = len(user_sorted)
        
        # Calculate split indices for this user
        train_end_idx = int(n * train_ratio)
        val_end_idx = int(n * (train_ratio + val_ratio))
        
        # Split this user's data
        train_list.append(user_sorted.iloc[:train_end_idx])
        val_list.append(user_sorted.iloc[train_end_idx:val_end_idx])
        test_list.append(user_sorted.iloc[val_end_idx:])
    
    # Combine all users' splits
    train_df = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
    test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()

    # Convert dates back to strings in YYYY-MM-DD format
    if not train_df.empty:
        train_df['date'] = train_df['date'].dt.strftime('%Y-%m-%d')
    if not val_df.empty:
        val_df['date'] = val_df['date'].dt.strftime('%Y-%m-%d')
    if not test_df.empty:
        test_df['date'] = test_df['date'].dt.strftime('%Y-%m-%d')

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Log split statistics
    logger.info(f"Train period: {train_df['date'].min()} to {train_df['date'].max()}")
    logger.info(f"Val period: {val_df['date'].min()} to {val_df['date'].max()}")
    logger.info(f"Test period: {test_df['date'].min()} to {test_df['date'].max()}")

    return train_df, val_df, test_df


def save_split_data(train_data: List[Dict], val_data: List[Dict], test_data: List[Dict], output_dir: str):
    """
    Save split data to JSON files.
    
    Args:
        train_data (List[Dict]): Training data
        val_data (List[Dict]): Validation data
        test_data (List[Dict]): Test data
        output_dir (str): Directory to save the files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each split
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f)
    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        json.dump(val_data, f)
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(test_data, f)
        
    logger.info(f"Saved split data to {output_dir}")

def load_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    file_path_str = str(file_path)  # Convert Path to string

    if file_path_str.endswith('.gz'):
        open_fn = gzip.open
        open_mode = 'rt'
    else:
        open_fn = open
        open_mode = 'r'

    with open_fn(file_path_str, open_mode, encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {os.path.basename(file_path_str)}"):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON line: {e}")
                continue
    return data

def process_review_data(review_data: List[Dict]) -> pd.DataFrame:
    """Process review data into a DataFrame."""
    processed_data = []
    for review in tqdm(review_data, desc="Processing review data"):
        processed_review = {
            'user_id': review.get('user_id'),
            'parent_asin': review.get('parent_asin'),
            'rating': review.get('rating'),
            'review_text': review.get('text', ''),
            'review_title': review.get('title', ''),
            'timestamp': review.get('timestamp'),
            #'verified_purchase': review.get('verified_purchase', False),
            #'helpful_votes': review.get('helpful_vote', 0)
        } # [user_id, parent_asin, rating, review_text, review_title, timestamp, verified_purchase, helpful_votes]
        processed_data.append(processed_review)
    return pd.DataFrame(processed_data)

def is_valid_image_url(url: str) -> bool:
    """Check if URL is valid and points to an image."""
    try:
        response = requests.head(url, timeout=5)
        content_type = response.headers.get('content-type', '')
        return response.status_code == 200 and 'image' in content_type.lower()
    except:
        return False

def download_image(url: str, cache_dir: Path, filename: str) -> Optional[str]:
    """Download and cache an image."""
    if not url:
        return None
        
    cache_path = cache_dir / filename
    
    # Check if image already exists in cache
    if cache_path.exists():
        try:
            # Verify the image is valid
            with Image.open(cache_path) as img:
                img.verify()
            return str(cache_path)
        except Exception as e:
            logger.warning(f"Found corrupted cached image {filename}, will redownload: {str(e)}")
            # Remove corrupted image
            cache_path.unlink()
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Save image
        with open(cache_path, 'wb') as f:
            f.write(response.content)
            
        # Verify the downloaded image
        with Image.open(cache_path) as img:
            img.verify()
            
        return str(cache_path)
    except Exception as e:
        logger.error(f"Error downloading image {url}: {str(e)}")
        if cache_path.exists():
            cache_path.unlink()
        return None

def process_metadata(metadata: List[Dict], cache_dir: Path) -> pd.DataFrame:
    """Process metadata into a DataFrame with image validation and caching."""
    processed_metadata = []
    invalid_images = []
    
    for item in tqdm(metadata, desc="Processing metadata"):
        # Get the main product image URL (highest quality)
        images = item.get('images', [])
        main_image_url = None
        local_image_path = None
        parent_asin = item.get('parent_asin')
        
        # Skip items without parent_asin
        if not parent_asin:
            logger.warning("Skipping item without parent_asin")
            continue
        
        for img in images:
            if img.get('variant') == 'MAIN':
                main_image_url = img.get('hi_res')
                break
        
        # Validate and cache image if URL exists
        if main_image_url:
            if is_valid_image_url(main_image_url):
                local_image_path = download_image(
                    main_image_url,
                    cache_dir,
                    f"{parent_asin}.jpg"
                )
            else:
                invalid_images.append({
                    'parent_asin': parent_asin,
                    'url': main_image_url
                })
        
        processed_item = {
            'parent_asin': parent_asin,
            'title': item.get('title', ''),
            #'description': ' '.join(item.get('description', [])),
            #'features': ' '.join(item.get('features', [])),
            #'price': item.get('price'),
            'main_category': item.get('main_category'),
            #'average_rating': item.get('average_rating'),
            #'rating_number': item.get('rating_number'),
            #'store': item.get('store'),
            #'brand': item.get('details', {}).get('Brand', ''),
            'image_url': main_image_url,
            'local_image_path': local_image_path
        }# ['parent_asin', 'title', 'price', 'main_category', 'store', 'brand', 'image_url', 'local_image_path']
        processed_metadata.append(processed_item)
    
    # Log invalid images 
    if invalid_images:
        logger.warning(f"Found {len(invalid_images)} invalid images")
        invalid_images_path = cache_dir.parent / "invalid_images.jsonl"
        with open(invalid_images_path, 'w') as f:
            for item in invalid_images:
                f.write(json.dumps(item) + '\n')
    
    return pd.DataFrame(processed_metadata)

def merge_data(review_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Merge review data with metadata using parent_asin."""
    logger.info("Merging review data with metadata...")
    merged_df = pd.merge(
        review_df,
        metadata_df,
        on='parent_asin',
        how='inner'
    )
    logger.info(f"Total records after merge: {len(merged_df)}")
    return merged_df


def save_processed_data(data: pd.DataFrame, output_path: Union[str, Path]):
    """Save processed data to JSONL file."""
    output_path_str = str(output_path)  # Convert Path to string
    logger.info(f"Saving processed data to {output_path_str}")
    data.to_json(output_path_str, orient='records', lines=True)
    logger.info("Data saved successfully")




# MULTIMODAL PROCESSOR:
# ==========================================

class ImageEncoder:
    def __init__(self, image_size: int = 224, vision_model_name: str = "clip-vit-b-32"):
        """
        Initialize the multi-modal processor for handling both text and image data.
        Args:
            image_size (int): Size to resize images to
            vision_model_name (str): Vision model name (clip-vit-b-32, resnet-18, mobilenet-v2, etc.)
        """
        self.image_size = image_size
        
        # Map model names to HuggingFace model identifiers
        model_mapping = {
            "clip-vit-b-32": "openai/clip-vit-base-patch32",
            "resnet-18": "microsoft/resnet-18",
            "resnet-50": "microsoft/resnet-50",
            "mobilenet-v2": "google/mobilenet_v2_1.0_224",
            "mobilenet-v3": "google/mobilenet_v3_small_100_224",
            "efficientnet-b0": "google/efficientnet-b0"
        }
        
        # Get the full model name
        if vision_model_name in model_mapping:
            full_model_name = model_mapping[vision_model_name]
        else:
            # Assume it's already a full HuggingFace model name
            full_model_name = vision_model_name
            
        logger.info(f"Loading vision model: {full_model_name} (alias: {vision_model_name})")
        try:
            # Initialize image processor and model with caching
            self.image_processor = AutoImageProcessor.from_pretrained(
                full_model_name, 
                use_fast=True,
                cache_dir="checkpoints"
            )
            self.vision_model = AutoModel.from_pretrained(
                full_model_name,
                cache_dir="checkpoints"
            )
            logger.info("Vision model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            raise

        # Keep this for backward compatibility; image_processor is used for model-specific preprocessing.
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        
            
    def process_image(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Process an image into a tensor suitable for the model.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Optional[torch.Tensor]: Processed image tensor if successful, None otherwise
        """
        try:
            # Use the HF image processor so each backbone (CLIP/ResNet/...) gets correct normalization.
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                processed = self.image_processor(images=img, return_tensors="pt")
                image_tensor = processed["pixel_values"].squeeze(0)
                if not isinstance(image_tensor, torch.Tensor):
                    logger.warning(f"Transform did not return a tensor for {image_path}")
                    return None
                return image_tensor
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None
            
    def get_image_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding for an image using the vision model.
        
        Args:
            image_tensor (torch.Tensor): Processed image tensor
            
        Returns:
            torch.Tensor: Image embedding
        """
        with torch.no_grad():
            outputs = self.vision_model(pixel_values=image_tensor.unsqueeze(0))

            # CLIPModel exposes semantically aligned embeddings directly.
            if hasattr(outputs, 'image_embeds') and outputs.image_embeds is not None:
                return outputs.image_embeds
            elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
                return pooled_output
            else:
                return outputs[0].mean(dim=1)


def custom_collate_fn(
    batch,
    pad_token_id=None,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
    ):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        
        # Flatten padded if it contains any lists
        flattened = []
        for x in padded:
            if isinstance(x, list):
                flattened.extend(x)
            else:
                flattened.append(x if x is not None else 0)
                
        inputs = torch.tensor(flattened[:-1], dtype=torch.long)  # Truncate the last token for inputs
        targets = torch.tensor(flattened[1:], dtype=torch.long)  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor