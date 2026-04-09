from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from datetime import datetime
from tqdm import tqdm
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

import src.data.processor as processor

class AmazonDatasetAlpaca(Dataset):
    # Adjust Dataset to Alpaca template
    def __init__(self, data, tokenizer, special_chars):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = processor.format_input_alpaca(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text, allowed_special=special_chars)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

#class AmazonDatasetPhi(Dataset):
#     # Class-level cache for image embeddings to avoid reprocessing
#     _image_cache = {}
    
#     # Adjust Dataset to Phi3 format with unified multimodal embedding
#     def __init__(self, data, tokenizer, special_chars, multimodal_encoder=None, vision_model_name="resnet-18"):
#         # Initialize multimodal encoder if not provided
#         if multimodal_encoder is None:
#             logger.info(f"Initializing ImageEncoder with vision model: {vision_model_name}")
#             try:
#                 multimodal_encoder = processor.ImageEncoder(vision_model_name=vision_model_name)
#                 logger.info("ImageEncoder initialized successfully")
#             except Exception as e:
#                 logger.error(f"Failed to initialize ImageEncoder: {e}")
#                 logger.warning("Continuing without image processing")
#                 multimodal_encoder = None
#         self.data = data
#         self.tokenizer = tokenizer
#         self.special_chars = set(special_chars)  # Convert to set
#         self.multimodal_encoder = multimodal_encoder
#         self.max_token_length = 1024  # Set this to your model's maximum input length

#         # Pre-process text and images into unified token sequences
#         self.encoded_sequences = []
#         logger.info(f"Processing {len(data)} entries for Phi3 dataset...")
        
#         for i, entry in enumerate(data):
#             if i % 1000 == 0:
#                 logger.info(f"Processing entry {i}/{len(data)}")
                
#             # Process text
#             instruction_plus_input = processor.format_input_phi(entry)
#             response_text = f"\n<|assistant|>:\n{entry['output']}"
            
#             # Get text tokens
#             text_tokens = self.tokenizer.encode(
#                 f"<|text|> {instruction_plus_input}",
#                 allowed_special=self.special_chars
#             )

#             # Process image if available and multimodal encoder is provided
#             if entry.get('local_image_path') and self.multimodal_encoder is not None:
#                 try:
#                     # Check if image is already cached
#                     image_path = entry['local_image_path']
#                     if image_path in self._image_cache:
#                         image_token_list = self._image_cache[image_path]
#                         logger.debug(f"Using cached image embedding for {image_path}")
#                     else:
#                         image_tensor = self.multimodal_encoder.process_image(image_path)
#                         if image_tensor is not None:
#                             # Get image tokens using vision encoder
#                             image_tokens = self.multimodal_encoder.get_image_embedding(image_tensor)
#                             # Convert image embeddings to token-like format (flatten and convert to integers)
#                             image_token_list = image_tokens.flatten().tolist()
#                             # Scale down image tokens to fit within tokenizer vocabulary
#                             image_token_list = [int(t * 1000) % 1000 for t in image_token_list[:100]]  # Limit to 100 tokens
#                             # Cache the processed image tokens
#                             self._image_cache[image_path] = image_token_list
#                         else:
#                             image_token_list = []
                    
#                     # Concatenate text and image tokens
#                     combined_tokens = text_tokens + image_token_list
#                 except Exception as e:
#                     logger.warning(f"Failed to process image for entry {i}: {e}")
#                     combined_tokens = text_tokens
#             else:
#                 combined_tokens = text_tokens

#             # Add response tokens at the end
#             response_tokens = self.tokenizer.encode(
#                 f"<|response|> {response_text}",
#                 allowed_special=self.special_chars
#             )
#             combined_tokens.extend(response_tokens)

#             # Check if the combined token length exceeds the model's context length
#             if len(combined_tokens) > self.max_token_length:
#                 combined_tokens = combined_tokens[:self.max_token_length]
            
#             self.encoded_sequences.append(combined_tokens)
        
#         logger.info(f"Finished processing {len(self.encoded_sequences)} sequences")
#         logger.info(f"Image cache size: {len(self._image_cache)} unique images")

#     def __getitem__(self, index):
#         return self.encoded_sequences[index]

#     def __len__(self):
#         return len(self.data)


class AmazonDatasetPhi(Dataset):
    # Class-level caches to avoid reprocessing
    _image_cache = {}
    _sequence_cache = {}
    
    # Adjust Dataset to Phi3 format with unified multimodal embedding
    def __init__(self, data, tokenizer, special_chars, multimodal_encoder=None, vision_model_name="resnet-18", config=None):
        # Check if image processing is enabled (default to True for multimodal)
        enable_image_processing = config.get('data_config', {}).get('enable_image_processing', True)
        if not enable_image_processing:
            logger.info("Image processing disabled in config")
        
        # Initialize multimodal encoder if not provided and enabled
        if multimodal_encoder is None and enable_image_processing:
            logger.info(f"Initializing ImageEncoder with vision model: {vision_model_name}")
            try:
                multimodal_encoder = processor.ImageEncoder(vision_model_name=vision_model_name)
                logger.info("ImageEncoder initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ImageEncoder: {e}")
                logger.warning("Continuing without image processing")
                multimodal_encoder = None
        elif not enable_image_processing:
            multimodal_encoder = None
            logger.info("Image processing disabled - using text-only mode")
        self.data = data
        self.tokenizer = tokenizer
        self.special_chars = set(special_chars)  # Convert to set
        self.multimodal_encoder = multimodal_encoder
        self.max_token_length = 1024  # Set this to your model's maximum input length

        # Pre-process text and images into unified token sequences
        self.encoded_sequences = []
        logger.info(f"Processing {len(data)} entries for Phi3 dataset...")
        
        # Count cached vs new entries
        cached_count = 0
        new_count = 0
        
        # Check if fast training mode is enabled
        fast_training = config.get('data_config', {}).get('fast_training_mode', False)
        
        for i, entry in enumerate(data):
            if i % 5000 == 0:  # Reduced logging frequency
                logger.info(f"Processing entry {i}/{len(data)}")
            
            # Create a unique key for this entry based on its content
            entry_key = self._create_entry_key(entry)
            
            # Check if entire sequence is already cached
            if entry_key in self._sequence_cache:
                combined_tokens = self._sequence_cache[entry_key]
                cached_count += 1
                logger.debug(f"Using cached sequence for entry {i}")
            else:
                # Process text
                instruction_plus_input = processor.format_input_phi(entry)
                
                # Use output from entry (should be present after proper processing)
                if 'output' in entry:
                    response_text = f"\n<|assistant|>:\n{entry['output']}"
                else:
                    # Fallback for backward compatibility
                    parent_asin = entry.get('parent_asin', '')
                    title = entry.get('title', '')
                    response_text = f"\n<|assistant|>:\nRecommended product: {title} (ID: {parent_asin})"
                
                # Get text tokens
                text_tokens = self.tokenizer.encode(
                    f"<|text|> {instruction_plus_input}",
                    allowed_special=self.special_chars
                )

                # Process image if available and multimodal encoder is provided
                image_path = entry.get('local_image_path')
                if image_path and self.multimodal_encoder is not None and not fast_training:
                    try:
                        # Check if image is already cached
                        if image_path in self._image_cache:
                            image_token_list = self._image_cache[image_path]
                        else:
                            # Process local image file
                            image_tensor = self.multimodal_encoder.process_image(image_path)
                            if image_tensor is not None:
                                # Get image tokens using vision encoder
                                image_tokens = self.multimodal_encoder.get_image_embedding(image_tensor)
                                # Convert image embeddings to token-like format (flatten and convert to integers)
                                image_token_list = image_tokens.flatten().tolist()
                                # Scale down image tokens to fit within tokenizer vocabulary
                                image_token_list = [int(t * 1000) % 1000 for t in image_token_list[:100]]  # Limit to 100 tokens
                                # Cache the processed image tokens
                                self._image_cache[image_path] = image_token_list
                            else:
                                image_token_list = []
                        
                        # Concatenate text and image tokens
                        combined_tokens = text_tokens + image_token_list
                    except Exception as e:
                        logger.warning(f"Failed to process image for entry {i}: {e}")
                        combined_tokens = text_tokens
                else:
                    # Skip image processing in fast training mode
                    combined_tokens = text_tokens

                # Add response tokens at the end
                response_tokens = self.tokenizer.encode(
                    f"<|response|> {response_text}",
                    allowed_special=self.special_chars
                )
                combined_tokens.extend(response_tokens)

                # Check if the combined token length exceeds the model's context length
                if len(combined_tokens) > self.max_token_length:
                    combined_tokens = combined_tokens[:self.max_token_length]
                
                # Cache the entire processed sequence
                self._sequence_cache[entry_key] = combined_tokens
                new_count += 1
            
            self.encoded_sequences.append(combined_tokens)
        
        logger.info(f"Finished processing {len(self.encoded_sequences)} sequences")
        logger.info(f"Image cache size: {len(self._image_cache)} unique images")
        logger.info(f"Sequence cache: {cached_count} cached, {new_count} new entries")
    
    def _create_entry_key(self, entry):
        """Create a unique key for an entry based on its content"""
        # Use a combination of text content and image path as the key
        text_content = f"{entry.get('input', '')}{entry.get('output', '')}"
        image_path = entry.get('local_image_path', '')
        return f"{hash(text_content)}_{hash(image_path)}"

    def __getitem__(self, index):
        return self.encoded_sequences[index]

    def __len__(self):
        return len(self.data)
