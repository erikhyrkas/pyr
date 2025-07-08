# Pyr 🔥

Pyr is a small, fast, efficient, and useful language model built on a standard architecture with only **135 million parameters**. It’s trained from scratch using a compact tokenizer, high-quality data, and following a training sequencing strategy.

This series of models is English only, which is unfortunate, but a compact tokenizer and limited training set is critical for the hardware I have and objectives of the model.

Use Pyr at your own risk.

---

## Goals

Many use cases require instruction following and speed, but do not require encyclopedic knowledge.

### Pyr Tokenizer (trained)

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

- A simple, standard model
- 16 layers, 768 hidden size, 12 attention heads, GQA with 4 kv-heads
- ReLU² activations, rotary position encoding
- Trained using `transformers.Trainer` with mixed-precision and aggressive batching
- Custom tokenizer: Trained on high-compression BPE from scratch
- ChatML prompt format (I personally dislike the format, but following the standard makes adoption easier.)

---

## Training Plan

### Pyr-base-135m

- Phase 1 (Cosmopedia-v2, 10B tokens @ 512 seq len)
- Phase 2 (RoyalRoad-1.61M filtered, 2.5B tokens @ 2048 seq len)
- Phase 3 (FineWeb-EDU, 10B tokens @ 1024 seq len)
- Phase 4 (Python-EDU corpus, 7B tokens @ 1024 seq len )

I noticed as phase 1 was winding down that the language was stilted, so I decided to do a round with a story-based dataset, but as there aren't many good ones and I didn't want to torrent books3, I decided to take a highly filtered version of Royal Road. The goal being to give the model a little more fluency before diving into facts. 

### Pyr-instruct-135m

- SFT (smol-smoltalk, 2b tokens @ 8192 seq len)

### Pyr-135m-reasoning (aspirational)

- SFT (reasoning-v1-20m, 35b tokens @ 8192 seq len)
- DPO/SFT (smoltalk, 8b tokens @ 8192 seq len)

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



