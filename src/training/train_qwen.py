import json
import logging
import re
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

from ..utils.utils import load_config, set_seed, setup_logging


setup_logging()
logger = logging.getLogger(__name__)


class PromptDataset(Dataset):
    def __init__(self, rows, include_output: bool):
        self.rows = rows
        self.include_output = include_output

    @staticmethod
    def build_prompt(entry: dict, include_output: bool) -> str:
        instruction = entry.get(
            "instruction",
            "Given a user's purchase history, predict the next 5 items they would purchase.",
        )
        input_text = entry.get("input", "")
        prompt = f"<|user|>\n{instruction}\n{input_text}<|end|>\n<|assistant|>\n"
        if include_output:
            prompt += entry.get("output", "")
        return prompt

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.build_prompt(self.rows[idx], include_output=self.include_output)


def build_collate_fn(tokenizer, max_length: int):
    def collate(batch_texts):
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        encoded["labels"] = labels
        return encoded

    return collate


def compute_val_loss(model, loader, device, max_batches: int = 20):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            total += out.loss.item()
            count += 1
    return total / max(count, 1)


def extract_recommendation_tokens(text: str) -> str:
    asin_pattern = r"<\|ASIN_[^|]+\|>"
    endoftext_pattern = r"<\|endoftext\|>"
    matches = re.findall(f"({asin_pattern}|{endoftext_pattern})", text)
    return ", ".join(matches)


def main():
    set_seed(42)

    config = load_config("configs/train_config_qwen.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path("data") / "processed"
    model_out_dir = Path("output") / "model" / "qwen2_5_7b_lora"
    model_out_dir.mkdir(parents=True, exist_ok=True)

    with (data_dir / "train.json").open("r", encoding="utf-8") as f:
        train_rows = json.load(f)
    with (data_dir / "val.json").open("r", encoding="utf-8") as f:
        val_rows = json.load(f)
    with (data_dir / "test.json").open("r", encoding="utf-8") as f:
        test_rows = json.load(f)
    with (data_dir / "special_user_item_ids.json").open("r", encoding="utf-8") as f:
        special_tokens = json.load(f)

    model_name = config["model_config"]["hf_model_name"]
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    user_special_tokens = [t for t in special_tokens if isinstance(t, str) and t.startswith("<|ASIN_")]
    extra_tokens = ["<|user|>", "<|assistant|>", "<|end|>", "<|text|>", "<|response|>"]
    all_added = list(dict.fromkeys(extra_tokens + user_special_tokens))
    tokenizer.add_special_tokens({"additional_special_tokens": all_added})

    use_4bit = bool(config["model_config"].get("use_4bit", False))
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto" if device.type == "cuda" else None,
        )

    model.resize_token_embeddings(len(tokenizer))

    lora_cfg = config["training_config"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg.get("lora_rank", 16)),
        lora_alpha=float(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=lora_cfg.get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model = get_peft_model(model, peft_config)
    model.train()
    model.gradient_checkpointing_enable()

    train_dataset = PromptDataset(train_rows, include_output=True)
    val_dataset = PromptDataset(val_rows, include_output=True)
    collate_fn = build_collate_fn(tokenizer, max_length=int(config["model_config"].get("max_length", 1024)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training_config"].get("batch_size", 1)),
        shuffle=True,
        num_workers=int(config["training_config"].get("num_workers", 2)),
        pin_memory=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    lr = float(config["training_config"].get("learning_rate", 2e-4))
    num_epochs = int(config["training_config"].get("num_epochs", 3))
    grad_accum = int(config["training_config"].get("gradient_accumulation_steps", 8))
    max_grad_norm = float(config["training_config"].get("max_grad_norm", 1.0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = max(1, (len(train_loader) * num_epochs) // max(1, grad_accum))
    warmup_steps = int(config["training_config"].get("warmup_steps", 50))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val = float("inf")
    patience = int(config["training_config"].get("early_stopping_patience", 3))
    no_improve = 0
    global_step = 0
    start = time.time()

    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Qwen epoch {epoch + 1}/{num_epochs}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / max(1, grad_accum)
            loss.backward()
            running += out.loss.item()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        val_loss = compute_val_loss(model, val_loader, device, max_batches=20)
        train_loss = running / max(1, len(train_loader))
        logger.info("Epoch %d: train_loss=%.4f val_loss=%.4f", epoch + 1, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            model.save_pretrained(model_out_dir)
            tokenizer.save_pretrained(model_out_dir)
            logger.info("Saved best adapter to %s", model_out_dir)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    logger.info("Qwen LoRA training finished in %.2f min", (time.time() - start) / 60)

    # Load best adapter before generation
    model.eval()
    gen_cfg = config.get("generation_config", {})
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 96))
    temperature = float(gen_cfg.get("temperature", 0.2))
    top_p = float(gen_cfg.get("top_p", 0.9))
    max_samples = int(gen_cfg.get("max_samples", 300))

    test_subset = test_rows if max_samples <= 0 else test_rows[:max_samples]
    for i, entry in enumerate(tqdm(test_subset, desc="Generating test responses with Qwen")):
        prompt = PromptDataset.build_prompt(entry, include_output=False)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(config["model_config"].get("max_length", 1024)))
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(out_ids[0], skip_special_tokens=False)
        response_text = decoded[len(prompt):] if decoded.startswith(prompt) else decoded
        test_rows[i]["model_response"] = extract_recommendation_tokens(response_text)

    if max_samples > 0 and max_samples < len(test_rows):
        for j in range(max_samples, len(test_rows)):
            test_rows[j]["model_response"] = ""

    out_file = data_dir / "test_with_responses.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(test_rows, f, indent=2)
    logger.info("Saved Qwen responses to %s", out_file)


if __name__ == "__main__":
    main()
