# Pyr 🔥

Pyr is a small, fast, efficient, and useful language model built on a custom BitNet 1.58-style architecture with only **135 million parameters**. It’s trained from scratch using a compact tokenizer, high-quality data, and following a training sequencing strategy.

This series of models is English only, which is unfortunate, but a compact tokenizer and limited training set is critical for the hardware I have and objectives of the model.

Use Pyr at your own risk.

---

## Goals

Many use cases require instruction following and speed, but require encyclopedic knowledge.

### Pyr Tokenizer (in progress)

- Compact (16k english vocab)
- Feature focused
- ChatML standard compliant
- Special handling for numbers (no bpe merging)
- Python and instructions captured in training

### Pyr-135m-base (in progress)

- Efficiency-first
- Coherent

### Pyr-135m-instruct (next up)

- Instruction following
- Multi-turn
- Tool calling 

### Pyr-135m-reasoning (aspirational)

- Reasoning

---

## Architecture

- Based on [BitNet 1.58](https://arxiv.org/abs/2504.12285v2), simplified for compact deployment
- 16 layers, 768 hidden size, 12 attention heads, GQA with 4 kv-heads
- ReLU² activations, rotary position encoding
- Trained using `transformers.Trainer` with mixed-precision and aggressive batching
- Custom tokenizer: Trained on high-compression BPE from scratch
- ChatML prompt format (I personally dislike the format, but following the standard makes adoption easier.)

---

## Training Plan

Pyr-base-135m:

- Phase 1 (Cosmopedia-v2, 10B tokens @ 512 seq len) 
- Phase 2 (FineWeb-EDU, 10B tokens @ 1024 seq len)
- Phase 3 (Python-EDU corpus)

Pyr-instruct-135m:

- SFT: smol-smoltalk  (484k rows/maybe 2b tokens)

Aspirational:

- SFT: reasoning-v1-20m (up to 22m rows/35b tokens, would likely take a sample)
- DPO/SFT: smoltalk (this is much bigger than smol-smoltalk...no that's not a typo or redudant.)

---

## Intended Use

Pyr is "use at your own risk."

* No guardrails 
* May not be factual
* Might not follow user intent

There is no guarantee, implied or otherwise.

---

## License

MIT License. Use, adapt, and build on it. Attribution welcome.



