import os
import torch
import random
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from pathlib import Path
import json

def setup_logging(log_dir: str = "logs") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_dir (str): Directory to store log files
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create and add file handler
    log_file = os.path.join(log_dir, "train.log")
    file_handler = logging.FileHandler(log_file, mode='a')  # 'a' for append mode
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create and add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Log a test message
    logger.info("Logging system initialized")

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    val_loss: float,
    checkpoint_path: Path
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        scheduler: The learning rate scheduler
        epoch: Current epoch number
        val_loss: Current validation loss
        checkpoint_path: Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss
    }
    
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path}")

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_path: Path
) -> Tuple[int, float]:
    """
    Load a training checkpoint.
    
    Args:
        model: The model to load state into
        optimizer: The optimizer to load state into
        scheduler: The scheduler to load state into
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Tuple of (epoch, val_loss)
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint['epoch'], checkpoint['val_loss']

def save_config(config: dict, config_path: Path) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f"Saved configuration to {config_path}")


def load_config(config_path: str = "configs/train_config.json") -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    logging.info(f"Loaded configuration from {config_path}")
    
    return config

def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics dictionary into a string.
    
    Args:
        metrics: Dictionary of metric names and values
        
    Returns:
        Formatted string of metrics
    """
    return " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

def create_directories(dirs: List[str]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        dirs: List of directory paths to create
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def get_device() -> torch.device:
    """
    Get the appropriate device for training.
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: Model to count parameters for
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 