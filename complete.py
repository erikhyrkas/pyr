import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def list_checkpoints(path):
    directories = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path) and entry.startswith("checkpoint-"):
            directories.append(entry)
    return directories


MODEL_DIR = "./pyr-135m-base-1"

if not os.path.exists(f"{MODEL_DIR}/config.json"):
    # get a list of directories in the path
    checkpoints = list_checkpoints(MODEL_DIR)
    checkpoint_to_use = None
    step_count = -1
    for entry in checkpoints:
        steps = int(entry.split("-")[1])
        if steps > step_count:
            checkpoint_to_use = entry
            step_count = steps
    if checkpoint_to_use is not None:
        print(f"Loading checkpoint: {checkpoint_to_use}")
        MODEL_DIR = f"{MODEL_DIR}/{checkpoint_to_use}"
    else:
        print("No checkpoints found.")
        exit(1)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

print("Type text to complete Ctrl+C to exit.")

try:
    while True:
        user_input = input("\nComplete: ")

        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

        generate_inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}

        with torch.no_grad():
            outputs = model.generate(
                **generate_inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.9,
                top_k=40,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Extract response (skip prompt)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(response)

except KeyboardInterrupt:
    print("\nGoodbye.")
