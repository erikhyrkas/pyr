from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from pathlib import Path

MAX_TOKENS = 8192
MIN_TOKENS = 4096
SAVE_PATH = "./clean_royalroad_chapters"
TOKENIZER_PATH = "./pyr-16k-tokenizer"
MIN_VIEWS = 1000

def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)

    excluded_authors = set()
    if Path("excluded_authors.txt").exists():
        with open("excluded_authors.txt", "r", encoding="utf-8") as f:
            excluded_authors = {line.strip() for line in f if line.strip()}

    print("Loading RoyalRoad dataset...")
    dataset = load_dataset("OmniAICreator/RoyalRoad-1.61M", split="train")

    clean_samples = []

    for i, entry in enumerate(dataset):
        author = entry.get("author")
        text = entry.get("text")
        chapter_title = entry.get("chapter_title", "")
        if chapter_title is None:
            chapter_title = "<untitled>"
        else:
            chapter_title = chapter_title.strip()
        avg_views = entry.get("average_views", 0)
        tags = set(entry.get("tags", []))
        warnings = set(entry.get("warnings", []))

        if (
            not author
            or not text
            or author in excluded_authors
            or avg_views < MIN_VIEWS
            or warnings.intersection({"Sexual Content", "Disturbing Content"})
            or tags.intersection({"Grimdark", "Harem"})
        ):
            continue

        full_text = f"Chapter title: {chapter_title}\n\n{text}" if chapter_title else text
        token_ids = tokenizer.encode(full_text, add_special_tokens=False)

        if len(token_ids) >= MIN_TOKENS:
            chunk = token_ids[:MAX_TOKENS]
            clean_samples.append({
                "input_ids": chunk,
                "length": len(chunk),
                "author": author,
            })

        if len(clean_samples) == 1 or ((len(clean_samples) % 10000) == 0 and len(clean_samples) > 0):
            print(f"Samples collected: {len(clean_samples):,}")

    # 159,771 total valid samples
    print(f"\nTotal valid samples: {len(clean_samples):,}")
    final_dataset = Dataset.from_list(clean_samples)

    split = final_dataset.train_test_split(test_size=0.01, seed=42)
    dataset_dict = DatasetDict({"train": split["train"], "test": split["test"]})

    dataset_dict.save_to_disk(SAVE_PATH)
    print(f"\n✅ Saved cleaned and tokenized dataset to {SAVE_PATH}")
    print(f"   Train size: {len(split['train'])}, Test size: {len(split['test'])}")


if __name__ == "__main__":
    main()
