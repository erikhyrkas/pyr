from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./pyr-16k-tokenizer", use_fast=True)

with open("excluded_authors.txt", "r", encoding="utf-8") as f:
    excluded_authors = set(line.strip() for line in f if line.strip())

MAX_TOKENS = 2048
OVERLAP = 512
MIN_TOKENS = 1024
SAVE_PATH = "./clean_royalroad_chunked"

print("Loading Royal Road dataset...")
dataset = load_dataset("OmniAICreator/RoyalRoad-1.61M", split="train")

clean_samples = []

for i, entry in enumerate(dataset):
    author = entry.get("author")
    text = entry.get("text")
    avg_views = entry.get("average_views", 0)
    tags = set(entry.get("tags", []))
    warnings = set(entry.get("warnings", []))

    if (
        not author
        or not text
        or author in excluded_authors
        or avg_views < 5000
        or warnings.intersection({"Sexual Content", "Disturbing Content"})
        or tags.intersection({"Grimdark", "Harem"})
    ):
        continue

    token_ids = tokenizer.encode(text, add_special_tokens=False)

    i = 0
    while i + MIN_TOKENS <= len(token_ids):
        chunk = token_ids[i : i + MAX_TOKENS]
        if len(chunk) >= MIN_TOKENS:
            clean_samples.append({
                "input_ids": chunk,
                "length": len(chunk),
                "author": author,
            })
        i += MAX_TOKENS - OVERLAP

    if (len(clean_samples) % 10000) < 50 and len(clean_samples) > 0:
        print(f"Chunks so far: {len(clean_samples):,}")

print(f"\nTotal valid samples: {len(clean_samples):,}")
dataset = Dataset.from_list(clean_samples)

split = dataset.train_test_split(test_size=0.01, seed=42)
final = DatasetDict({"train": split["train"], "test": split["test"]})

final.save_to_disk(SAVE_PATH)
print(f"\n✅ Saved cleaned and tokenized dataset to {SAVE_PATH}")
print(f"   Train size: {len(split['train'])}, Test size: {len(split['test'])}")
