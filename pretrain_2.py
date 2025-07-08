import os
import torch
import gc
from datasets import load_from_disk
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from prepare_clean_story_dataset import SAVE_PATH

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

def causal_collator(features):
    # Remove non-token fields like 'author', 'length'
    features = [{"input_ids": f["input_ids"]} for f in features]

    # Pad and return as input + labels
    padded = tokenizer.pad(
        features,
        padding=True,
        return_tensors="pt"
    )
    padded["labels"] = padded["input_ids"].clone()
    return padded


def main():
    print("=" * 70)
    print("PYR PRETRAINING Phase 2 - Narrative Fluency")
    print("=" * 70)

    DATASET_PATH = SAVE_PATH
    PHASE1_MODEL_PATH = "./pyr-135m-base-1/checkpoint-128000"
    OUTPUT_DIR = "./pyr-135m-base-2"

    EPOCHS = 1
    MAX_LENGTH = 2048  # Informational only here

    print("Loading tokenizer from Phase 1...")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PHASE1_MODEL_PATH, use_fast=True)

    print("Loading model from Phase 1...")
    model = LlamaForCausalLM.from_pretrained(
        PHASE1_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("Model and tokenizer loaded.")
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.1f}M")

    print(f"\nLoading filtered RoyalRoad dataset from disk: {DATASET_PATH}")
    dataset = load_from_disk(DATASET_PATH)

    print(f"Dataset loaded:")
    print(f"   Train: {len(dataset['train']):,} samples")
    print(f"   Eval:  {len(dataset['test']):,} samples")
    gc.collect()

    batch_size = 4
    grad_accum = 32
    effective_batch_size = batch_size * grad_accum

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=EPOCHS,
        save_total_limit=2,
        logging_steps=100,
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=1000,
        bf16=True,
        max_grad_norm=1.0,
        logging_dir=f"{OUTPUT_DIR}/logs",
        remove_unused_columns=False,
        report_to=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=causal_collator
    )

    print("\n" + "=" * 70)
    print("STARTING PYR PRETRAINING PHASE 2")
    print("=" * 70)

    try:
        trainer.train()

        print("\nTRAINING COMPLETE!")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        trainer.save_state()

        with open(f"{OUTPUT_DIR}/training_info.txt", "w") as f:
            f.write(f"Pyr 135m Base - Phase 2 (RoyalRoad filtered)\n")
            f.write(f"Parameters: {total_params:.1f}M\n")
            f.write(f"Training samples: {len(dataset['train']):,}\n")
            f.write(f"Training sequence length: {MAX_LENGTH}\n")
            f.write(f"Effective batch size: {effective_batch_size}\n")
            f.write(f"Final eval loss: {trainer.state.log_history[-1].get('eval_loss', 'N/A')}\n")
            f.write(f"Continued from: {PHASE1_MODEL_PATH}\n")

        print(f"Model saved to: {OUTPUT_DIR}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        model.save_pretrained(f"{OUTPUT_DIR}-interrupted")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}-interrupted")
        trainer.save_state()

    except Exception as e:
        print(f"\nTraining failed: {e}")
        model.save_pretrained(f"{OUTPUT_DIR}-failed")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}-failed")
        trainer.save_state()
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("PHASE 2 PRETRAINING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
