import torch
from functools import partial
from pathlib import Path
import json
import logging
import tiktoken
from torch.utils.data import DataLoader
import os
import urllib.request
import time
from tqdm import tqdm
import re
import multiprocessing
import torch.multiprocessing as mp
import torch.cuda.amp as amp
from torch import autocast
import numpy as np
import gc
import sys

def setup_cuda():
    """Setup CUDA with proper memory allocation settings."""
    if torch.cuda.is_available():
        # Fix the multiprocessing issue
        torch.cuda.memory._set_allocator_settings('expandable_segments:False')
        
        # Clear any existing cache
        torch.cuda.empty_cache()
        
        # Set memory fraction to avoid OOM (more conservative)
        torch.cuda.set_per_process_memory_fraction(0.6)
        
        print(f"CUDA setup complete:")
        print(f"   - Device: {torch.cuda.get_device_name(0)}")
        print(f"   - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   - Expandable segments: False")
        return True
    else:
        print("CUDA not available, using CPU")
        return False

# Fix CUDA multiprocessing issue properly
if torch.cuda.is_available():
    torch.cuda.memory._set_allocator_settings('expandable_segments:False')
    print(f"CUDA available: {torch.cuda.device_count()} devices")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available, using CPU")

from transformers.models.gpt2 import GPT2Model

from ..models.model import assign_check, load_weights, train_model

# Set multiprocessing start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

# Enable gradient checkpointing
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def clear_cuda_cache():
    """Clear CUDA cache to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# CHANGES:
# - model/ calc_loss_batch
# model / GPTModel, generate

from ..data.processor import custom_collate_fn, format_input_phi, format_input_alpaca, ImageEncoder
from ..data.dataset import AmazonDatasetPhi, AmazonDatasetAlpaca
from ..utils.utils import get_device, load_config, setup_logging, set_seed
from ..models import model as model_fn
from ..utils.visualization import plot_cross_entropy_loss, plot_losses_over_tokens_seen


setup_logging()
logger = logging.getLogger(__name__)
set_seed(42)


def main():
    # Setup CUDA properly first
    use_cuda = setup_cuda()
    
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    cache_dir = processed_dir / "image_cache"

    model_dir = Path("output") / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoints directory for vision model caching
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()

    # Potential error: These files might not exist
    try:
        #end_of_text_id = json.load(open(processed_dir / "end_of_text_id.json"))
        special_user_item_ids =  set(["<|endoftext|>"] + json.load(open(processed_dir / "special_user_item_ids.json")))
    except FileNotFoundError as e:
        logger.error(f"Required JSON file not found: {e}")
        raise

    # 1.1 Set the tokenizers
    #_______________________________________________________________________________
    try:
        base_tokenizer = tiktoken.get_encoding("gpt2")
        custom_token_ids = {token: base_tokenizer.n_vocab + i for i, token in enumerate(special_user_item_ids)}
        tokenizer = tiktoken.Encoding(
        name="gpt2_custom",
        pat_str=base_tokenizer._pat_str,
        mergeable_ranks=base_tokenizer._mergeable_ranks,
        special_tokens={**base_tokenizer._special_tokens, **custom_token_ids},
        )

        # Add special tokens to the tokenizer
        end_of_text_id = np.array(tokenizer.encode("<|endoftext|>", allowed_special=special_user_item_ids)).item()


    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

    # =================================================================================
    # 1.0 Set custom collate function
    # =================================================================================
    device = str(get_device())
    
    # Potential error: These files might not exist
    try:
        train_data = json.load(open(processed_dir / "train.json"))
        val_data = json.load(open(processed_dir / "val.json"))
        test_data = json.load(open(processed_dir / "test.json"))
    except FileNotFoundError as e:
        logger.error(f"Required data file not found: {e}")
        raise

    logger.info("Finished loading train, val and test data")

    customized_collate_fn = partial(custom_collate_fn, pad_token_id=end_of_text_id, 
                                    allowed_max_length=config['model_config']['context_length'], 
                                    device=device)
    
    logger.info("Finished setting custom collate function")
    phi3_prompt = config['custom_config']['phi3']

    if phi3_prompt:
        CustomDataset = AmazonDatasetPhi
    else:
        CustomDataset = AmazonDatasetAlpaca

    # Get vision model configuration
    vision_model_name = config['custom_config'].get('vision_model', 'resnet-18')
    logger.info(f"Using vision model: {vision_model_name}")

    # Create a single multimodal encoder to share between datasets
    multimodal_encoder = None
    if phi3_prompt:
        logger.info(f"Initializing shared ImageEncoder with vision model: {vision_model_name}")
        try:
            multimodal_encoder = ImageEncoder(vision_model_name=vision_model_name)
            logger.info("Shared ImageEncoder initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ImageEncoder: {e}")
            logger.warning("Continuing without image processing")
            multimodal_encoder = None

    logger.info("Creating training dataset...")
    train_dataset = CustomDataset(train_data, tokenizer, special_user_item_ids, 
                                 multimodal_encoder=multimodal_encoder, config=config)
    logger.info(f"Training dataset created with {len(train_dataset)} samples")
    
    logger.info("Creating validation dataset...")
    val_dataset = CustomDataset(val_data, tokenizer, special_user_item_ids, 
                               multimodal_encoder=multimodal_encoder, config=config)
    logger.info(f"Validation dataset created with {len(val_dataset)} samples")


    # Set the data loaders
    logger.info("Creating training data loader...")
    train_loader = DataLoader(
        train_dataset, batch_size=config['data_config']['batch_size'], 
        collate_fn=customized_collate_fn, drop_last=True,
        shuffle=False, num_workers=config['data_config']['num_workers'])
    logger.info("Training data loader created")
    
    logger.info("Creating validation data loader...")
    val_loader = DataLoader(
        val_dataset, batch_size=config['data_config']['batch_size'], 
        collate_fn=customized_collate_fn, drop_last=True,
        shuffle=False, num_workers=config['data_config']['num_workers'])
    logger.info("Validation data loader created")
    
    logger.info("Finished setting the data loaders")

    # 1.2 Load pretrained model
    #_______________________________________________________________________________
    logger.info("Setting up model configuration...")
    BASE_CONFIG = {
        "vocab_size": config['model_config']['vocab_size'],     # Vocabulary size
        "context_length": config['model_config']['context_length'],  # Context length
        "drop_rate": config['model_config']['drop_rate'],        # Dropout rate
        "qkv_bias": config['model_config']['qkv_bias']         # Query-key-value bias
    }

    model_configs = config['gpt_model_config']
    CHOOSE_MODEL = config['model_config']['model_name']

    # Potential error: Model name might not be in configs
    if CHOOSE_MODEL not in model_configs:
        raise ValueError(f"Model {CHOOSE_MODEL} not found in model_configs")

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    #________________________
    # allowed model names
    model_names = {
        "gpt2-small (124M)": "openai-community/gpt2",
        "gpt2-medium (355M)": "openai-community/gpt2-medium",
        "gpt2-large (774M)": "openai-community/gpt2-large",
        "gpt2-xl (1558M)": "openai-community/gpt2-xl"
    }

    logger.info(f"Loading HuggingFace model: {model_names[CHOOSE_MODEL]}")
    model_hf = GPT2Model.from_pretrained(model_names[CHOOSE_MODEL], cache_dir="checkpoints")
    model_hf.eval()
    logger.info("HuggingFace model loaded successfully")

    logger.info("Creating custom GPT model...")
    model = model_fn.GPTModel(BASE_CONFIG)
    load_weights(model, model_hf, BASE_CONFIG)
    logger.info("Weights loaded successfully")
    logger.info(f"Moving model to device: {device}")
    model.to(device)
    
    logger.info(f"Finished loading the weights: {CHOOSE_MODEL}")

    # Replace the old embedding layer with the new one in the model
    #________________________________________________________________

    #logging.info("Update the token embedding layer")
    num_tokens, emb_size = model.tok_emb.weight.shape
    new_num_tokens = num_tokens + len(special_user_item_ids)

    # Create a new embedding layer and move it to the same device as the model
    new_embedding = torch.nn.Embedding(new_num_tokens, emb_size).to(device)

    # Copy weights from the old embedding layer
    new_embedding.weight.data[:num_tokens] = model.tok_emb.weight.data

    # Replace the old embedding layer with the new one in the model
    model.tok_emb = new_embedding
    #logger.info("Finished replacing the old embedding layer with the new one in the model")
    logging.info("Updated the token embedding layer")

    # Update the output layer
    #________________________________________________________________
    original_out_features, original_in_features = model.out_head.weight.shape

    # Define the new number of output features
    new_out_features = original_out_features + len(special_user_item_ids)

    # Create a new linear layer with the extended output size and move it to the same device
    new_linear = torch.nn.Linear(original_in_features, new_out_features).to(device)


    # Copy the weights and biases from the original linear layer
    with torch.no_grad():
        new_linear.weight[:original_out_features] = model.out_head.weight
        if model.out_head.bias is not None:
            new_linear.bias[:original_out_features] = model.out_head.bias

    # Replace the original linear layer with the new one
    model.out_head = new_linear
    logging.info("Updated the output layer")


    # LORA

    lora = config['custom_config']['lora']

    if lora:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters before: {total_params:,}")

        for param in model.parameters():
            param.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters after: {total_params:,}")
        model_fn.replace_linear_with_lora(model, rank=16, alpha=16)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total trainable LoRA parameters: {total_params:,}")
        model.to(device)

    # =================================================================================
    # 2: Start finetuning the model
    # =================================================================================
    logging.info("Starting finetuning the model")

    # Initial Loss
    #_______________________________________________________________________________
    logger.info("Initial losses")
    with torch.no_grad():
        train_loss = model_fn.calc_loss_loader(train_loader, model, device, num_batches=4)
        val_loss = model_fn.calc_loss_loader(val_loader, model, device, num_batches=4)

    logger.info(f"   Training loss: {train_loss}")
    logger.info(f"   Validation loss: {val_loss}")
    logger.info(f"   Percentage change: {round(100 * (val_loss - train_loss)/train_loss, 2)}%")

    # Training
    #_______________________________________________________________________________
    start_time = time.time()

    num_epochs = int(config['training_config']['num_epochs'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training_config']['learning_rate'], weight_decay=config['training_config']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    start_context = format_input_phi(val_data[0]) if phi3_prompt else format_input_alpaca(val_data[0])

    #train_losses, val_losses, tokens_seen = model_fn.train_model_simple(
    #    model, train_loader, val_loader, optimizer, device,
    #    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    #    start_context=start_context, tokenizer=tokenizer,
    #    special_chars=special_user_item_ids,
    #)

    # Set up checkpoint path
    checkpoint_path = model_dir / "training_checkpoint.pth"

    # Use our optimized training function instead of train_model_simple
    model, train_losses, val_losses, tokens_seen = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        logger=logger,
        start_context=start_context,
        tokenizer=tokenizer,
        special_chars=special_user_item_ids,
        checkpoint_path=str(checkpoint_path)
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    logger.info(f"> Training completed in {execution_time_minutes:.2f} minutes.")

    # Terminate execution here for testing
    #sys.exit(0)


    logger.info("Plotting losses")
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_name = "loss-plot.pdf"
    if phi3_prompt:
        plot_name_ce = plot_name.replace(".pdf", "-phi3-prompt.pdf")
        plot_name_tokens = plot_name_ce.replace(".pdf", "-phi3-prompt-tokens.pdf")
    if lora:
        plot_name_ce = plot_name.replace(".pdf", "-lora.pdf")
        plot_name_tokens = plot_name_ce.replace(".pdf", "-lora-tokens.pdf")
    if not any([phi3_prompt, lora]):
        plot_name_ce = plot_name.replace(".pdf", "-baseline.pdf")
        plot_name_tokens = plot_name_ce.replace(".pdf", "-baseline-tokens.pdf")

    plot_dir = Path("output") / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path_ce = plot_dir / plot_name_ce
    plot_path_tokens = plot_dir / plot_name_tokens

    try:
        plot_cross_entropy_loss(train_losses, val_losses, save_path=plot_path_ce)
        plot_losses_over_tokens_seen(epochs_tensor, tokens_seen, train_losses, val_losses, save_path=plot_path_tokens)
        logger.info(f"> Saved plots to {plot_path_ce} and {plot_path_tokens}")
    except Exception as e:
        logger.error(f"Failed to create plots: {e}")

    # =================================================================================
    # 3: Generate responses
    # =================================================================================
    skip_response_generation = config['training_config'].get('skip_response_generation', False)
    if skip_response_generation:
        logger.info("Skipping response generation (skip_response_generation=true)")
        test_data_path = processed_dir / "test_with_responses.json"
        for entry in test_data:
            entry["model_response"] = ""
        with open(test_data_path, "w") as file:
            json.dump(test_data, file, indent=4)
        logger.info(f"Saved empty response placeholders to {test_data_path}")
    else:
        generation_max_samples = int(config['training_config'].get('generation_max_samples', 0) or 0)
        generation_max_new_tokens = int(
            config['training_config'].get(
                'generation_max_new_tokens',
                config['training_config']['max_new_tokens']
            )
        )

        if generation_max_samples > 0:
            logger.info(
                f"Fast generation enabled: processing first {generation_max_samples} "
                f"of {len(test_data)} test samples"
            )
            test_generation_data = test_data[:generation_max_samples]
        else:
            test_generation_data = test_data

        for i, entry in tqdm(enumerate(test_generation_data), total=len(test_generation_data), desc="Generating responses"):
            try:
                input_text = format_input_phi(entry) if phi3_prompt else format_input_alpaca(entry)
            
            # Convert input text to token ids and move to device
                input_ids = model_fn.text_to_token_ids(input_text, tokenizer, special_user_item_ids).to(device)
            
            # Generate response with adjusted parameters
                token_ids = model_fn.generate(
                    model=model,
                    idx=input_ids,
                    max_new_tokens=generation_max_new_tokens,
                    context_size=BASE_CONFIG["context_length"],
                    eos_id=end_of_text_id,
                    temperature=config['training_config']['temperature'],
                    top_k=config['training_config']['top_k'], 
                    top_p=config['training_config']['top_p']
                )
            
            # Convert generated tokens back to text
                generated_text = model_fn.token_ids_to_text(token_ids, tokenizer)
            
            # Extract only the response part
                if phi3_prompt:
                    if "<|assistant|>" in generated_text:
                        response_text = generated_text.split("<|assistant|>")[-1].strip()
                    else:
                        response_text = generated_text[len(input_text):].strip()
                else:
                    response_text = generated_text[len(input_text):].split("### Response:")[-1].strip()
            
            # Clean up the response
            #response_text = re.findall(r'<\|ASIN_[^|]+\|>', response_text)
            #response_text = ', '.join(response_text)

            # Extract ASINs or endoftext token
                asin_pattern = r'<\|ASIN_[^|]+\|>'
                endoftext_pattern = r'<\|endoftext\|>'
                matches = re.findall(f'({asin_pattern}|{endoftext_pattern})', response_text)
                response_text = ', '.join(matches)
            
            #response_text = response_text.replace("<|endoftext|>", "").strip()
            # Remove any trailing punctuation or special characters
            #response_text = re.sub(r'[!]+$', '', response_text)
            
            # Store the response
                test_data[i]["model_response"] = response_text
            
            # Log occasional samples
                if i % 100 == 0:
                    logger.info(f"\nSample generation {i}:")
                    logger.info(f"Input: {input_text[:100]}...")
                    logger.info(f"Response: {response_text}")

            except Exception as e:
                logger.error(f"Failed to generate response for entry {i}: {e}")
                test_data[i]["model_response"] = ""

        if generation_max_samples > 0 and generation_max_samples < len(test_data):
            # Mark remaining entries as unanswered to keep output format stable.
            for i in range(generation_max_samples, len(test_data)):
                test_data[i]["model_response"] = ""

    # =================================================================================
    # 4: Save the model 
    # =================================================================================
    try:
        model_path = model_dir /  f"finetuned-{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    # =================================================================================
    # 5: Save the test data with responses
    # =================================================================================
    try:
        test_data_path = processed_dir / "test_with_responses.json"
        with open(test_data_path, "w") as file:
            json.dump(test_data, file, indent=4)
        logger.info(f"Saved test data with responses to {test_data_path}")
    except Exception as e:
        logger.error(f"Failed to save test data responses: {e}")


# functions: assign_check, load_weights, train_model




if __name__ == "__main__":
    main()