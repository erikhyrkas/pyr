import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from complete import list_checkpoints

MODEL_DIR = "./pyr-135m-instruct"

# -----------------------------------------------------------------------------
# Locate the newest checkpoint (if fine‑tuning is still in progress)
# -----------------------------------------------------------------------------
if not os.path.exists(f"{MODEL_DIR}/config.json"):
    checkpoints = list_checkpoints(MODEL_DIR)
    latest_step = -1
    latest_ckpt = None
    for ckpt in checkpoints:
        try:
            steps = int(ckpt.split("-")[1])
        except (IndexError, ValueError):
            continue
        if steps > latest_step:
            latest_step = steps
            latest_ckpt = ckpt

    if latest_ckpt is None:
        print("No checkpoints found – aborting.")
        raise SystemExit(1)

    MODEL_DIR = f"{MODEL_DIR}/{latest_ckpt}"
    print(f"Loading latest checkpoint: {latest_ckpt}")

# -----------------------------------------------------------------------------
# Tokenizer / Model
# -----------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

# Device placement
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dtype).to(DEVICE)
model.eval()
MAX_CTX = model.config.max_position_embeddings

# -----------------------------------------------------------------------------
# Conversation helpers
# -----------------------------------------------------------------------------

def build_chat_prompt(messages):
    """Return a prompt string using the tokenizer's chat template when available."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        # add_generation_prompt=True appends the assistant tag so the model knows to continue
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Fallback to a minimal ChatML‑style format
    formatted = "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
    )
    formatted += "<|im_start|>assistant\n"  # Generation starts here
    return formatted

# Seed the conversation with a system directive; adjust to taste
conversation = [
    {"role": "system", "content": "You are Pyr, a helpful AI assistant."},
]

print("🔥 Pyr is ready. Press Ctrl+C to exit.")

try:
    while True:
        user_input = input("\n> ").strip()
        if not user_input:
            continue

        # Add user turn to conversation
        conversation.append({"role": "user", "content": user_input})

        # Build prompt and ensure it fits into model context
        while True:
            prompt = build_chat_prompt(conversation)
            toks = tokenizer(prompt, return_tensors="pt")
            if toks["input_ids"].shape[1] <= MAX_CTX or len(conversation) <= 2:
                break
            # Drop the oldest non‑system message and retry
            conversation.pop(1)
        toks = toks.to(DEVICE)
        gen_inputs = {k: v for k, v in toks.items() if k != "token_type_ids"}

        with torch.no_grad():
            output = model.generate(
                **gen_inputs,
                max_new_tokens=256,
                temperature=0.8,
                top_k=40,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        reply = decoded[len(prompt):].strip()
        print(f"Pyr: {reply}")

        # Add assistant turn so future context is preserved
        conversation.append({"role": "assistant", "content": reply})

        # Trim conversation if it grows too large (keep system + last 39 messages)
        if len(conversation) > 40:
            conversation = [conversation[0]] + conversation[-39:]

except KeyboardInterrupt:
    print("\n🔥 Goodbye.")
