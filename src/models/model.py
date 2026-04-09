#_____________________________________________________________________________________________
# LIBRARY PACKAGES
#_____________________________________________________________________________________________
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import autocast
from torch.cuda.amp import GradScaler
import gc
import math
import logging
import torch.cuda.amp as amp
import os

logger = logging.getLogger(__name__)

class ImprovedCrossAttention(nn.Module):
    """Enhanced cross-attention for better multimodal fusion."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text_features, image_features, attention_mask=None):
        """Cross-attention between text and image features."""
        attended_text, _ = self.cross_attention(
            text_features, image_features, image_features,
            key_padding_mask=attention_mask
        )
        # Residual connection and layer norm
        output = self.norm(text_features + self.dropout(attended_text))
        return output

class ProgressivePredictionHead(nn.Module):
    """Progressive prediction heads for item and stop prediction."""
    def __init__(self, d_model, vocab_size, dropout=0.1):
        super().__init__()
        self.item_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, vocab_size)
        )
        self.stop_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)  # Binary: continue/stop
        )
        
    def forward(self, hidden_states):
        item_logits = self.item_head(hidden_states)
        stop_logits = self.stop_head(hidden_states)
        return item_logits, stop_logits

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        
        # Enhanced multimodal components
        self.use_improved_multimodal = cfg.get("use_improved_multimodal", False)
        if self.use_improved_multimodal:
            # Cross-attention for multimodal fusion
            self.cross_attention = ImprovedCrossAttention(
                d_model=cfg["emb_dim"],
                num_heads=cfg.get("num_heads", 8),
                dropout=cfg.get("dropout", 0.1)
            )
            
            # Progressive prediction heads
            self.progressive_head = ProgressivePredictionHead(
                d_model=cfg["emb_dim"],
                vocab_size=cfg["vocab_size"],
                dropout=cfg.get("dropout", 0.1)
            )
            
            # Image encoder for better image processing
            self.image_encoder = nn.Sequential(
                nn.Linear(cfg.get("image_embedding_dim", 2048), 1024),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(1024, cfg["emb_dim"]),
                nn.LayerNorm(cfg["emb_dim"])
            )
        else:
            # Original output head
            self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
        # Enable gradient checkpointing by default
        self.gradient_checkpointing = False

        # Enable memory efficient attention
        self.use_memory_efficient_attention = True

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def forward(self, in_idx, image_embeddings=None, attention_mask=None, return_multimodal_outputs=False):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        
        # Process image embeddings if available and using improved multimodal
        if self.use_improved_multimodal and image_embeddings is not None:
            # Encode image embeddings to match text dimension
            image_features = self.image_encoder(image_embeddings)
            
            # Apply cross-attention between text and images
            x = self.cross_attention(x, image_features, attention_mask)
        
        # Apply gradient checkpointing if enabled
        if self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(inputs[0])
                return custom_forward
            
            # Process each transformer block with checkpointing
            for block in self.trf_blocks:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    use_reentrant=False,  # More stable but slightly slower
                    preserve_rng_state=False  # Save memory by not preserving RNG state
                )
        else:
            x = self.trf_blocks(x)
            
        x = self.final_norm(x)
        
        # Generate outputs based on model configuration
        if self.use_improved_multimodal and return_multimodal_outputs:
            # Use progressive prediction heads
            item_logits, stop_logits = self.progressive_head(x)
            return {
                'item_logits': item_logits,
                'stop_logits': stop_logits,
                'hidden_states': x
            }
        else:
            # Use original output head
            logits = self.out_head(x)
            return logits


        
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def generate(model, idx, max_new_tokens, context_size, eos_id=None, temperature=1.0, top_k=None, top_p=1.0, repetition_penalty=1.0):
    """Generate text with improved sampling parameters."""
    model.eval()
    for _ in range(max_new_tokens):
        # Crop idx to the last context_size tokens
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step
        logits = logits[:, -1, :] / temperature
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for i in range(idx.size(1)):
                logits[:, idx[0, i]] /= repetition_penalty
        
        # Apply top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Apply nucleus sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Stop if we predict the end of sequence token
        if eos_id is not None and idx_next.item() == eos_id:
            break
    
    return idx


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, special_chars=None
                       ):
    # First version of the train model
    # Downside: It is slow and consumes lots of memory

    scaler = GradScaler(device)  # Initialize the gradient scaler for mixed precision
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            # Use autocast to reduce memory usage with mixed precision
            if device == "cuda":
                with autocast(device_type="cuda"):
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
            else:
                 with autocast(device_type="cpu"):
                    loss = calc_loss_batch(input_batch, target_batch, model, device)


            # Scale the loss and call backward() with the scaler
            scaler.scale(loss).backward()

            # Update model weights using the scaler
            scaler.step(optimizer)
            scaler.update()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(
            model, tokenizer, device, start_context, special_chars
        )

        # Clear GPU cache after each epoch
        if device == "cuda":
            torch.cuda.empty_cache()

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context, special_chars=None):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer, special_chars).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def text_to_token_ids(text, tokenizer, special_chars):
    encoded = tokenizer.encode(text, allowed_special=special_chars)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """Calculate loss for a batch of data with weighted stop prediction."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    # Forward pass
    logits = model(input_batch)

    # Check for NaN or Inf values in inputs and outputs
    if torch.isnan(input_batch).any() or torch.isinf(input_batch).any():
        logger.error("NaN or Inf detected in input batch")
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logger.error("NaN or Inf detected in logits")
    
    # Get EOS token ID (assuming it's the last token in vocab)
    eos_token_id = logits.shape[-1] - 1  # Adjust based on your tokenizer
    
    # Create weights for loss calculation
    # Give higher weight to EOS predictions to improve stop prediction
    weights = torch.ones(logits.shape[-1], device=device)
    weights[eos_token_id] = 2.0  # Double weight for EOS token
    
    # Calculate weighted loss
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten(),
        weight=weights
    )

    # Check for NaN or Inf values in loss
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        logger.error("NaN or Inf detected in loss")
    
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            #print(f"Batch {i} - Input shape: {input_batch.shape}, Target shape: {target_batch.shape}")
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches




# MODELS:
def assign_check(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach())

def load_weights(gpt, gpt_hf, BASE_CONFIG):

    d = gpt_hf.state_dict()

    gpt.pos_emb.weight = assign_check(gpt.pos_emb.weight, d["wpe.weight"])
    gpt.tok_emb.weight = assign_check(gpt.tok_emb.weight, d["wte.weight"])
    
    for b in range(BASE_CONFIG["n_layers"]):
        q_w, k_w, v_w = np.split(d[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign_check(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign_check(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign_check(gpt.trf_blocks[b].att.W_value.weight, v_w.T)
    
        q_b, k_b, v_b = np.split(d[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign_check(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign_check(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign_check(gpt.trf_blocks[b].att.W_value.bias, v_b)
    
    
        gpt.trf_blocks[b].att.out_proj.weight = assign_check(gpt.trf_blocks[b].att.out_proj.weight, d[f"h.{b}.attn.c_proj.weight"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign_check(gpt.trf_blocks[b].att.out_proj.bias, d[f"h.{b}.attn.c_proj.bias"])
    
        gpt.trf_blocks[b].ff.layers[0].weight = assign_check(gpt.trf_blocks[b].ff.layers[0].weight, d[f"h.{b}.mlp.c_fc.weight"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign_check(gpt.trf_blocks[b].ff.layers[0].bias, d[f"h.{b}.mlp.c_fc.bias"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign_check(gpt.trf_blocks[b].ff.layers[2].weight, d[f"h.{b}.mlp.c_proj.weight"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign_check(gpt.trf_blocks[b].ff.layers[2].bias, d[f"h.{b}.mlp.c_proj.bias"])
    
        gpt.trf_blocks[b].norm1.scale = assign_check(gpt.trf_blocks[b].norm1.scale, d[f"h.{b}.ln_1.weight"])
        gpt.trf_blocks[b].norm1.shift = assign_check(gpt.trf_blocks[b].norm1.shift, d[f"h.{b}.ln_1.bias"])
        gpt.trf_blocks[b].norm2.scale = assign_check(gpt.trf_blocks[b].norm2.scale, d[f"h.{b}.ln_2.weight"])
        gpt.trf_blocks[b].norm2.shift = assign_check(gpt.trf_blocks[b].norm2.shift, d[f"h.{b}.ln_2.bias"])
    
        # 28-07-2025: Currently inside the loop but maybe should be outside
    
    gpt.final_norm.scale = assign_check(gpt.final_norm.scale, d["ln_f.weight"])
    gpt.final_norm.shift = assign_check(gpt.final_norm.shift, d["ln_f.bias"])
    gpt.out_head.weight = assign_check(gpt.out_head.weight, d["wte.weight"])


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, config, logger, start_context=None, tokenizer=None, special_chars=None, checkpoint_path=None):
        """Train the model with memory optimizations and tracking"""
        # Initialize mixed precision training
        scaler = amp.GradScaler()
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1
        start_epoch = 0

        # Checkpoint loading
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            try:
                # Load model state
                model.load_state_dict(checkpoint['model_state_dict'])

                # Load optimizer state
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # Load scheduler state
                if scheduler and 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                # Load scaler state
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])

                # Load training state
                train_losses = checkpoint.get('train_losses', [])
                val_losses = checkpoint.get('val_losses', [])
                track_tokens_seen = checkpoint.get('track_tokens_seen', [])
                tokens_seen = checkpoint.get('tokens_seen', 0)
                global_step = checkpoint.get('global_step', -1)
                start_epoch = checkpoint.get('epoch', 0)

                logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")
                logger.info(f"Previous losses - Train: {train_losses[-1] if train_losses else 'N/A'}, Val: {val_losses[-1] if val_losses else 'N/A'}")

            except RuntimeError as e:
                # Typical case: vocab/token changes alter embedding/output shapes.
                logger.warning(
                    "Checkpoint is incompatible with current model shape. "
                    "Starting training from scratch and ignoring checkpoint."
                )
                logger.warning(f"Checkpoint load error: {e}")

        # Enable gradient checkpointing at model level
        model.gradient_checkpointing_enable()
        
        # Set environment variable for better memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster computation
        torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmarking

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = config['training_config'].get('early_stopping_patience', 3)
        min_delta = config['training_config'].get('early_stopping_min_delta', 0.001)
        
        for epoch in range(start_epoch, config['training_config']['num_epochs']):
            model.train()
            
            for input_batch, target_batch in train_loader:
                # Clear CUDA cache periodically
                #if global_step % 5 == 0:
                #    torch.cuda.empty_cache()
                #    gc.collect()

                if torch.isnan(input_batch).any() or torch.isinf(input_batch).any():
                    logger.warning("NaN or Inf detected in input batch, skipping this batch.")
                    continue  # Skip this batch

                optimizer.zero_grad()
                
                # Use autocast to reduce memory usage with mixed precision
                if device == "cuda":
                    with autocast(device_type="cuda"):
                        # Check for NaN or Inf in input batch
                        if torch.isnan(input_batch).any() or torch.isinf(input_batch).any():
                            logger.error("NaN or Inf detected in input batch")
                            continue
                        # Forward pass
                        outputs = calc_loss_batch(input_batch, target_batch, model, device)

                        # Add detailed logging for debugging
                        #logger.debug(f"Batch input IDs: {input_batch}")
                        #logger.debug(f"Batch attention mask: {target_batch}")
                        #logger.debug(f"Batch labels: {target_batch}")
                        #logger.debug(f"Batch image embeddings: {input_batch}")

                        # Check for NaN or Inf in model's output (logits)
                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                            logger.error("NaN or Inf detected in model's output (logits), skipping this batch.")
                            continue  # Skip this batch

                        # Calculate loss
                        loss = outputs
                else:
                    with autocast(device_type="cpu"):
                        if torch.isnan(input_batch).any() or torch.isinf(input_batch).any():
                            logger.error("NaN or Inf detected in input batch")
                            continue
                        # Forward pass
                        outputs = calc_loss_batch(input_batch, target_batch, model, device)

                        # Check for NaN or Inf in model's output (logits)
                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                            logger.error("NaN or Inf detected in model's output (logits), skipping this batch.")
                            continue  # Skip this batch

                        # Calculate loss
                        loss = outputs

                # Scale the loss and call backward() with the scaler
                scaler.scale(loss).backward()

                # Clip gradients to prevent exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update model weights using the scaler
                scaler.step(optimizer)
                scaler.update()
                
                # Clear CUDA cache periodically to prevent OOM
                if global_step % 10 == 0:
                    torch.cuda.empty_cache()
                
                tokens_seen += input_batch.numel()
                global_step += 1
                
                # Check max training steps limit
                max_steps = config['training_config'].get('max_training_steps', 1000)
                if global_step >= max_steps:
                    logger.info(f"Reached maximum training steps ({max_steps}), stopping training")
                    return model, train_losses, val_losses, track_tokens_seen
                
                if global_step % config['training_config']['eval_freq'] == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, config['training_config']['eval_iter'])
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    logger.info(f"Ep {epoch+1} (Step {global_step:06d}): "
                            f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                    
                    # Early stopping check
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                        logger.info(f"New best validation loss: {val_loss:.3f}")
                    else:
                        patience_counter += 1
                        logger.info(f"No improvement for {patience_counter} evaluations")
                        
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"Early stopping triggered after {patience_counter} evaluations without improvement")
                            logger.info(f"Best validation loss: {best_val_loss:.3f}")
                            return model, train_losses, val_losses, track_tokens_seen
                    
                    # Save checkpoint
                    if checkpoint_path:
                        checkpoint = {
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'scaler_state_dict': scaler.state_dict(),
                            'train_losses': train_losses,
                            'val_losses': val_losses,
                            'track_tokens_seen': track_tokens_seen,
                            'tokens_seen': tokens_seen,
                            'config': config
                        }
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Checkpoint saved to {checkpoint_path}")
                    
                    # Update learning rate using scheduler
                    if scheduler:
                        scheduler.step(val_loss)
            
            # Generate sample after each epoch
            if start_context and tokenizer:
                generate_and_print_sample(
                    model, tokenizer, device, start_context, special_chars
                )
            
            # Clear GPU cache after each epoch
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
        
        return model, train_losses, val_losses, track_tokens_seen



class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # similar to standard weight initialization
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Replace the Linear layer with LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively apply the same function to child modules
            replace_linear_with_lora(module, rank, alpha)