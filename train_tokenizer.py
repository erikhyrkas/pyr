"""
Pyr-16k Byte‑Level BPE tokenizer
================================
* ChatML minimal tokens → `<|im_start|>` & `<|im_end|>` only.
* Atomic digits 0‑9 (digits are special tokens; cannot be merged).
* Spaces folded into the leading byte (Ġ) for good compression.
* 16,384 vocab, HuggingFace *tokenizers* (Rust backend).
* Multi‑char operators (`==`, `!=`, `>=`, …) pre‑seeded so they never split.
"""
import json
import os
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from tokenizers import (
    Tokenizer, models, pre_tokenizers, trainers, decoders,
    normalizers
)
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# ───────────────────────── CONFIG ─────────────────────────
DATASET_NAME   = "HuggingFaceTB/smol-smoltalk"
VOCAB_SIZE     = 16_384 # 24_576 # 32_768
OUTPUT_DIR     = "./pyr-16k-tokenizer"
NUM_SAMPLES    = 8_000_000   # streamed rows
MODEL_MAX_LEN  = 8_192

# ChatML v0.1 control symbols (minimal) --------------------
CHATML_TOKENS = ["<|im_start|>", "<|im_end|>"]
# Model housekeeping tokens --------------------------------
HOUSE_TOKENS  = ["<bos>", "<eos>", "<pad>", "<unk>"]
# Multi‑char operators -------------------------------------
OPERATORS     = [
    "==", "!=", ">=", "<=", "+=", "-=", "*=", "/=",
    "//", "**", "->", "=>"
]
# Decimal digits -------------------------------------------
DIGITS = list("0123456789")

# I suspect I shouldn't have put OPERATORS + DIGITS here:
# I've historically made tokenizers from scratch or used pre-rolled tokenizers and didn't remember this would be an issue with decoding later where these tokens would be invisible.
SPECIAL_TOKENS = CHATML_TOKENS + HOUSE_TOKENS + OPERATORS + DIGITS

# ──────────────────── DATASET ITERATOR ────────────────────

def chatml_iterator(ds, limit: int) -> Iterable[str]:
    """Yield each conversation in minimal ChatML format."""
    for i, sample in enumerate(ds):
        if i >= limit:
            break
        pieces: list[str] = []
        for msg in sample["messages"]:
            role = msg["role"]
            content = msg["content"].rstrip()
            pieces.append(f"<|im_start|>{role}\n{content}\n<|im_end|>\n")
        yield "".join(pieces)

# ─────────────────── TOKENIZER TRAINING ───────────────────

def train_pyr16k_tokenizer():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("*** Training Pyr‑16k tokenizer (ByteLevel BPE) ***")

    # 1 · Stream dataset (lazy)
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    # 2 · Skeleton
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.Sequence([normalizers.NFC()])
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=False),
        pre_tokenizers.Digits(individual_digits=True),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = ByteLevelProcessor(trim_offsets=True)

    # 3 · Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        initial_alphabet=DIGITS,  # ensures digits always exist as base tokens
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # 4 · Train
    tok.train_from_iterator(chatml_iterator(ds, NUM_SAMPLES), trainer)

    # 5 · Save tokenizer.json
    raw_path = Path(OUTPUT_DIR) / "tokenizer.json"
    tok.save(str(raw_path))

    # 6 · HuggingFace wrapper ----------------------------------------------
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=CHATML_TOKENS + OPERATORS + DIGITS,
        model_max_length=MODEL_MAX_LEN,
    )

    # Minimal chat template
    hf_tok.chat_template = (
        "{% for m in messages %}"
        "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>"
        "{% endfor %}"
        "<|im_start|>assistant\n"
    )

    hf_tok.save_pretrained(OUTPUT_DIR)

    # 7 · Self‑tests ---------------------------------------------------------
    sample = "Year 2025 → 3.14 * 2 = 6.28"
    encoded = hf_tok.encode(sample)

    # a) digits atomic
    for tid in encoded:
        tok_txt = hf_tok.decode([tid])
        assert not (tok_txt.isdigit() and len(tok_txt) > 1), "digit merge bug"

    # b) operator intact
    assert hf_tok.tokenize("x!=y")[1] == "!=", "operator split bug"

    # c) ChatML tokens survive round‑trip
    test_chat = "<|im_start|>user\nHello<|im_end|>"
    assert hf_tok.decode(hf_tok.encode(test_chat)) == test_chat, "ChatML corruption"

    print("✔ All self‑tests passed")

    # 8 · Quick reference vs GPT‑2 -----------------------------------------
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    for name, tknzr in (("Pyr‑16k", hf_tok), ("GPT‑2", gpt2_tok)):
        print(f"{name:8}: {len(tknzr.encode(sample))} tokens — vocab {tknzr.vocab_size:,}")

    # 9 · Stats file --------------------------------------------------------
    stats = {
        "tokenizer_type": "ByteLevel‑BPE",
        "vocab_size": hf_tok.vocab_size,
        "dataset": DATASET_NAME,
        "samples": NUM_SAMPLES,
        "special_tokens": {s: hf_tok.convert_tokens_to_ids(s) for s in SPECIAL_TOKENS},
    }
    with open(Path(OUTPUT_DIR) / "tokenizer_stats.json", "w") as fp:
        json.dump(stats, fp, indent=2)

    return hf_tok

# ────────────────────────── MAIN ──────────────────────────
if __name__ == "__main__":
    try:
        train_pyr16k_tokenizer()
    except KeyboardInterrupt:
        print("Interrupted by user")
