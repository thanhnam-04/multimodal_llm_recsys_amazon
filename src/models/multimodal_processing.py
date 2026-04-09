import os
import json
import requests
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModalProcessor:
    def __init__(
        self, 
        image_size: int = 224, 
        cache_dir: str = "data/image_cache",
        vision_model_name: str = "microsoft/resnet-50"
    ):
        """Initialize the multi-modal processor for handling product data.
        
        Args:
            image_size: Size to resize images to
            cache_dir: Directory to cache downloaded images
            vision_model_name: Name of the vision model to use
        """
        self.image_size = image_size
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize image processor and model
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name)
        self.vision_model = AutoModel.from_pretrained(vision_model_name)
        
        # Image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def process_product_data(self, product_data: Dict) -> Dict:
        """Process product metadata including images if available.
        
        Args:
            product_data: Dictionary containing product information
            
        Returns:
            Dictionary with processed product data
        """
        processed_data = {
            'title': product_data.get('title', ''),
            'category': product_data.get('category', ''),
            'price': product_data.get('price', ''),
            'rating': product_data.get('rating', ''),
            'review_count': product_data.get('review_count', ''),
            'timestamp': product_data.get('timestamp', ''),
            'features': product_data.get('features', []),
            'description': product_data.get('description', '')
        }
        
        # Process image if available
        if 'image_url' in product_data:
            image_path = self.download_image(product_data['image_url'])
            if image_path:
                image_tensor = self.process_image(image_path)
                if image_tensor is not None:
                    processed_data['image_embedding'] = self.get_image_embedding(image_tensor)
        
        return processed_data

    def process_sequence(
        self, 
        sequence_data: List[Dict]
    ) -> Tuple[List[Dict], Dict]:
        """Process a sequence of product interactions.
        
        Args:
            sequence_data: List of product interactions in temporal order
            
        Returns:
            Tuple containing:
            - List of processed product data
            - Temporal features dictionary
        """
        processed_sequence = []
        
        # Process each product in sequence
        for product in sequence_data:
            processed_data = self.process_product_data(product)
            processed_sequence.append(processed_data)
            
        # Extract temporal features
        temporal_features = self._get_temporal_features(sequence_data)
            
        return processed_sequence, temporal_features

    def _get_temporal_features(self, sequence_data: List[Dict]) -> Dict:
        """Extract temporal patterns from purchase history.
        
        Args:
            sequence_data: List of product interactions
            
        Returns:
            Dictionary containing temporal features
        """
        timestamps = [item['timestamp'] for item in sequence_data]
        time_diffs = np.diff(timestamps)
        
        # Get purchase patterns
        dates = [datetime.fromtimestamp(ts) for ts in timestamps]
        weekdays = [d.weekday() for d in dates]
        hours = [d.hour for d in dates]
        
        return {
            'avg_time_between_purchases': np.mean(time_diffs) if len(time_diffs) > 0 else 0,
            'std_time_between_purchases': np.std(time_diffs) if len(time_diffs) > 0 else 0,
            'common_purchase_days': pd.Series(weekdays).mode().tolist() if weekdays else [],
            'common_purchase_hours': pd.Series(hours).mode().tolist() if hours else [],
            'total_purchases': len(sequence_data),
            'first_purchase': min(timestamps) if timestamps else None,
            'last_purchase': max(timestamps) if timestamps else None
        }

    def download_image(self, url: str) -> Optional[str]:
        """Download an image from URL and save to cache."""
        try:
            filename = os.path.join(self.cache_dir, url.split('/')[-1])
            
            if os.path.exists(filename):
                return filename
                
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                f.write(response.content)
                
            return filename
            
        except Exception as e:
            logger.warning(f"Failed to download image from {url}: {str(e)}")
            return None
            
    def process_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Process an image into a tensor suitable for the model."""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor
            
        except Exception as e:
            logger.warning(f"Failed to process image {image_path}: {str(e)}")
            return None
            
    def get_image_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Get the embedding for an image using the vision model."""
        with torch.no_grad():
            outputs = self.vision_model(image_tensor.unsqueeze(0))
            return outputs.last_hidden_state.mean(dim=1)