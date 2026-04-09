import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import json
import logging


logger = logging.getLogger(__name__)

def plot_cross_entropy_loss(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[Path] = None,
    loss_type: str = "Cross Entropy",
    figsize: tuple = (9, 6)
) -> None:
    """
    Plot training and validation losses.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Optional path to save the plot
        loss_type: Type of loss being plotted (e.g., "Cross Entropy", "MSE", etc.)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot losses
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    
    # Customize plot
    plt.title(f"{loss_type} Loss During Training")
    plt.xlabel('Epoch')
    plt.ylabel(f'{loss_type} Loss')
    plt.legend()
    plt.grid(True)
    
    # Add annotations for best validation loss
    best_val_epoch = np.argmin(val_losses)
    best_val_loss = val_losses[best_val_epoch]
    plt.annotate(
        f'Best Val Loss: {best_val_loss:.4f}',
        xy=(best_val_epoch, best_val_loss),
        xytext=(10, 10),
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
    )
    
    # Save plot if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved loss plot to {save_path}")
    
    plt.close()


def plot_losses_over_tokens_seen(epochs_seen, tokens_seen, train_losses, val_losses, save_path: Optional[Path] = None):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved loss plot to {save_path}")
    plt.show()



def plot_metrics(
    metrics: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    title: str = "Model Metrics",
    figsize: tuple = (12, 6)
) -> None:
    """
    Plot multiple metrics over time.
    
    Args:
        metrics: Dictionary of metric names and their values over time
        save_path: Optional path to save the plot
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name, marker='o')
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved metrics plot to {save_path}")
    
    plt.close()

def save_training_history(
    history: Dict[str, List[float]],
    save_path: Path
) -> None:
    """
    Save training history to a JSON file.
    
    Args:
        history: Dictionary containing training metrics
        save_path: Path to save the history
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=4)
    logger.info(f"Saved training history to {save_path}")

def load_training_history(load_path: Path) -> Dict[str, List[float]]:
    """
    Load training history from a JSON file.
    
    Args:
        load_path: Path to load the history from
        
    Returns:
        Dictionary containing training metrics
    """
    with open(load_path, 'r') as f:
        history = json.load(f)
    logger.info(f"Loaded training history from {load_path}")
    return history 