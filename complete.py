import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, humanize, time

def check_quantization_status(model, max_layers=5):
    """Check quantization status of multiple layers"""
    print("\nQuantization Status:")
    layer_count = 0
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:
            unique_vals = torch.unique(param.data.abs())
            print(f"  {name}: {len(unique_vals):,} unique absolute values")
            if len(unique_vals) <= 10:
                values = unique_vals.cpu().numpy()
                print(f"    Values: {values}")
            else:
                print(f"    Range: [{unique_vals.min():.6f}, {unique_vals.max():.6f}]")

            layer_count += 1
            if layer_count >= max_layers:
                break


def gpu_mem():
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()  # live tensors
    reserved = torch.cuda.memory_reserved()  # allocator bucket
    peak = torch.cuda.max_memory_allocated()  # since last reset
    print(f"alloc={humanize.naturalsize(alloc, binary=True):>9}  "
          f"reserved={humanize.naturalsize(reserved, binary=True):>9}  "
          f"peak={humanize.naturalsize(peak, binary=True):>9}")


def list_checkpoints(path):
    directories = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path) and entry.startswith("checkpoint-"):
            directories.append(entry)
    return directories


def main():
    phase_paths = ["./pyr-135m-base-4", "./pyr-135m-base-3", "./pyr-135m-base-2", "./pyr-135m-base-1"]
    for path in phase_paths:
        if os.path.exists(path):
            MODEL_DIR = path
            if not os.path.exists(f"{MODEL_DIR}/config.json"):
                checkpoints = list_checkpoints(MODEL_DIR)
                checkpoint_to_use = None
                step_count = -1
                for entry in checkpoints:
                    steps = int(entry.split("-")[1])
                    if steps > step_count:
                        checkpoint_to_use = entry
                        step_count = steps
                if checkpoint_to_use is not None:
                    print(f"Model: {MODEL_DIR}")
                    print(f"Loading checkpoint: {checkpoint_to_use}")
                    MODEL_DIR = f"{MODEL_DIR}/{checkpoint_to_use}"
                elif path == "./pyr-135m-base-1":
                    print("No checkpoints found.")
                    return
            break

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("./pyr-16k-tokenizer", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"model.config.use_cache = {model.config.use_cache}")
    model.eval()

    gpu_mem()
    check_quantization_status(model)

    print(f"\n{'=' * 50}")
    print("TESTING GENERATION")
    print("=" * 50)
    print("Type text to complete (Ctrl+C to exit)")

    try:
        while True:
            user_input = input("\nComplete: ")

            inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
            generate_inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}

            start = time.time()
            MAX_TOKENS = 512
            # sampling is helpful to see more sane outputs early in training
            # eventually, the greedy decoding is faster and good enough.
            with torch.no_grad():
                outputs = model.generate(
                    **generate_inputs,
                    max_new_tokens=MAX_TOKENS,
                    # do_sample=False,
                    do_sample=True,
                    top_p=0.98,
                    top_k=40,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            end = time.time()
            elapsed = end - start
            print(f"Response: {response}")
            gpu_mem()
            generated_len = outputs[0].shape[0] - inputs["input_ids"].shape[1]
            print(f"Tokens/sec: {generated_len / elapsed:.2f}")

    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
