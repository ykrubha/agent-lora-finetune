"""LoRA fine-tuning with a small, explicit PyTorch training loop.

Only the assistant's turns count toward the loss. The system prompt, question and tool results
are context, because the model's job is to learn what to do with them, not to reproduce them.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE = -100


def encode_example(example: dict, tokenizer, max_length: int) -> dict | None:
    """Tokenize a conversation, marking assistant turns as the training targets.

    Returns None if it's too long. Raises ValueError if the chat template doesn't render
    conversations consistently, because the labels would then be wrong without any error.
    """
    messages, tools = example["messages"], example["tools"]

    def render(items: list[dict], prompt: bool) -> str:
        return tokenizer.apply_chat_template(items, tools=tools, tokenize=False, add_generation_prompt=prompt)

    full = render(messages, False)
    spans = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        start_text = render(messages[:index], True)
        end_text = render(messages[:index + 1], False)
        if not full.startswith(start_text) or not end_text.startswith(start_text) or not full.startswith(end_text):
            raise ValueError(f"The chat template isn't prefix-consistent at message {index}.")
        spans.append((len(start_text), len(end_text)))

    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    if len(ids) > max_length:
        return None
    labels = [IGNORE] * len(ids)
    for position, (start, end) in enumerate(encoded["offset_mapping"]):
        if any(low <= start and end <= high and end > start for low, high in spans):
            labels[position] = ids[position]
    return {"input_ids": ids, "labels": labels}


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def encode_all(examples: list[dict], tokenizer, max_length: int) -> tuple[list[dict], int]:
    encoded = [encode_example(example, tokenizer, max_length) for example in examples]
    kept = [item for item in encoded if item is not None]
    return kept, len(encoded) - len(kept)


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    ids = [torch.tensor(item["input_ids"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]
    input_ids = pad_sequence(ids, batch_first=True, padding_value=pad_id)
    return {
        "input_ids": input_ids,
        "attention_mask": pad_sequence([torch.ones_like(i) for i in ids], batch_first=True, padding_value=0),
        "labels": pad_sequence(labels, batch_first=True, padding_value=IGNORE),
    }


def learning_rate(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def validation_loss(model, data: list[dict], pad_id: int, device: str, batch_size: int, use_amp: bool) -> float:
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(data), batch_size):
        batch = collate(data[start:start + batch_size], pad_id)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            loss = model(**batch).loss
        tokens = int((batch["labels"] != IGNORE).sum())
        total += float(loss) * tokens
        count += tokens
    model.train()
    return total / max(count, 1)


def train(
    model_name: str,
    train_path: str | Path,
    validation_path: str | Path,
    output_dir: str | Path,
    epochs: int = 3,
    batch_size: int = 2,
    accumulation: int = 4,
    peak_lr: float = 2e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    max_length: int = 3072,
    seed: int = 7,
) -> dict:
    """Train and keep the adapter with the lowest validation loss. Restarts resume by epoch."""
    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    output_dir = Path(output_dir)
    latest, best = output_dir / "latest", output_dir / "best"
    state_path = output_dir / "state.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    train_data, dropped_train = encode_all(load_jsonl(train_path), tokenizer, max_length)
    val_data, dropped_val = encode_all(load_jsonl(validation_path), tokenizer, max_length)
    if not train_data or not val_data:
        raise ValueError("No usable examples. Increase max_length.")

    dtype = torch.float16 if use_amp else torch.float32
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()

    state = {"epoch": 0, "step": 0, "best_val": None, "history": []}
    if state_path.exists() and latest.exists():
        state = json.loads(state_path.read_text())
        model = PeftModel.from_pretrained(base, latest, is_trainable=True)
        print(f"Resuming after epoch {state['epoch']}.", flush=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    everything = sum(p.numel() for p in model.parameters())
    print(f"Training {trainable:,} of {everything:,} parameters on {device}. "
          f"{len(train_data)} train and {len(val_data)} validation examples "
          f"({dropped_train + dropped_val} too long and dropped).", flush=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=peak_lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    batches_per_epoch = math.ceil(len(train_data) / batch_size)
    total_steps = max(1, epochs * batches_per_epoch // accumulation)
    warmup = max(1, total_steps // 20)
    step = state["step"]
    started = time.time()

    model.train()
    for epoch in range(state["epoch"], epochs):
        order = list(range(len(train_data)))
        random.Random(seed + epoch).shuffle(order)
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for index, start in enumerate(range(0, len(order), batch_size)):
            batch = collate([train_data[i] for i in order[start:start + batch_size]], pad_id)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss / accumulation).backward()
            running += float(loss.detach())
            finished = index + 1 == batches_per_epoch
            if (index + 1) % accumulation == 0 or finished:
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(step, total_steps, peak_lr, warmup)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch + 1} step {step}/{total_steps} loss {running / (index + 1):.4f} "
                          f"{time.time() - started:.0f}s", flush=True)

        val_loss = validation_loss(model, val_data, pad_id, device, batch_size, use_amp)
        state["history"].append({"epoch": epoch + 1, "train_loss": round(running / batches_per_epoch, 4),
                                 "validation_loss": round(val_loss, 4)})
        print(state["history"][-1], flush=True)
        model.save_pretrained(latest)
        if state["best_val"] is None or val_loss < state["best_val"]:
            state["best_val"] = val_loss
            model.save_pretrained(best)
            tokenizer.save_pretrained(best)
        state.update(epoch=epoch + 1, step=step)
        state_path.write_text(json.dumps(state, indent=1))

    return {"train_examples": len(train_data), "validation_examples": len(val_data),
            "dropped": dropped_train + dropped_val, "trainable_parameters": trainable,
            "total_parameters": everything, "seconds_this_run": round(time.time() - started),
            "device": device, "model": model_name, **state}
