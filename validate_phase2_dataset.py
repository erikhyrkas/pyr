import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
import os

def get_latest_checkpoint(output_dir):
    checkpoints = [
        os.path.join(output_dir, d) for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {output_dir}")
    return sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]

MODEL_DIR = get_latest_checkpoint("./pyr-135m-base-2")
TOKENIZER_DIR = "./pyr-16k-tokenizer"
DATASET_PATH = "./clean_royalroad_chapters"
BATCH_SIZE = 8
NUM_SAMPLES = 64  # adjust to increase coverage

def main():
    print("\n🔍 Loading tokenizer and model...")
    print(f"📍 Using checkpoint: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()

    assert tokenizer.vocab_size == model.config.vocab_size, (
        f"❌ Vocab size mismatch! Tokenizer: {tokenizer.vocab_size}, Model: {model.config.vocab_size}"
    )
    print("✅ Tokenizer and model vocab sizes match.")

    print(f"\n📦 Loading dataset from {DATASET_PATH}...")
    dataset = load_from_disk(DATASET_PATH)
    print(f"   Train: {len(dataset['train'])} samples, Test: {len(dataset['test'])} samples")

    print("\n🧪 Decoding sample chunks:")
    for i in random.sample(range(len(dataset["train"])), 5):
        ids = dataset["train"][i]["input_ids"]
        print(f"\nSample {i} (length {len(ids)}):\n" + tokenizer.decode(ids))

    print("\n📊 Computing basic statistics...")
    subset = dataset["train"].select(range(1000))
    lengths = [len(x["input_ids"]) for x in subset]
    print(f" - Avg token length: {np.mean(lengths):.2f}")
    print(f" - Max token length: {np.max(lengths)}")
    print(f" - Min token length: {np.min(lengths)}")
    short_count = sum(1 for l in lengths if l < 100)
    print(f" - Very short (<100 tokens): {short_count}")

    print("\n🧠 Evaluating loss on a small batch...")
    sampled = random.sample(list(dataset["test"]), NUM_SAMPLES)
    input_ids = [torch.tensor(s["input_ids"])[:2048] for s in sampled]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss.item()

    print(f"\n📉 Sample loss over {NUM_SAMPLES} examples: {loss:.4f} (Perplexity ≈ {np.exp(loss):.2f})")

if __name__ == "__main__":
    main()
