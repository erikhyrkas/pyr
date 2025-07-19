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

def get_last_checkpoint(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    checkpoints = [
        os.path.join(output_dir, d) for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    return sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]

def causal_collator(features):
    features = [{"input_ids": f["input_ids"]} for f in features]
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
    PHASE1_DIR = "./pyr-135m-base-1"
    OUTPUT_DIR = "./pyr-135m-base-2"
    LAST_PHASE1_CHECKPOINT = get_last_checkpoint(PHASE1_DIR)

    if not LAST_PHASE1_CHECKPOINT:
        raise RuntimeError(f"No checkpoint found in {PHASE1_DIR}!")

    print(f"Loading tokenizer from: {LAST_PHASE1_CHECKPOINT}")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LAST_PHASE1_CHECKPOINT, use_fast=True)

    print(f"Loading model from: {LAST_PHASE1_CHECKPOINT}")
    model = LlamaForCausalLM.from_pretrained(
        LAST_PHASE1_CHECKPOINT,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.1f}M")

    print(f"\nLoading filtered RoyalRoad dataset from disk: {DATASET_PATH}")
    dataset = load_from_disk(DATASET_PATH)
    print(f"   Train: {len(dataset['train']):,} samples")
    print(f"   Eval:  {len(dataset['test']):,} samples")
    gc.collect()

    EPOCHS = 3
    batch_size = 2
    grad_accum = 64
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
        learning_rate=2e-5,  # very low final LR
        weight_decay=0.01,
        warmup_steps=0,
        bf16=True,
        max_grad_norm=1.0,
        logging_dir=f"{OUTPUT_DIR}/logs",
        remove_unused_columns=False,
        report_to=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
        lr_scheduler_type="constant",  # minimal scheduler effect
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
        trainer.train()  # don't resume a phase 2 checkpoint, always fresh

        print("\nTRAINING COMPLETE!")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        trainer.save_state()

        with open(f"{OUTPUT_DIR}/training_info.txt", "w") as f:
            f.write(f"Pyr 135m Base - Phase 2 (RoyalRoad filtered)\n")
            f.write(f"Parameters: {total_params:.1f}M\n")
            f.write(f"Training samples: {len(dataset['train']):,}\n")
            f.write(f"Training sequence length: 8192\n")
            f.write(f"Effective batch size: {effective_batch_size}\n")
            f.write(f"Final eval loss: {trainer.state.log_history[-1].get('eval_loss', 'N/A')}\n")
            f.write(f"Continued from: {LAST_PHASE1_CHECKPOINT}\n")

        print(f"Model saved to: {OUTPUT_DIR}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer.save_model(f"{OUTPUT_DIR}-interrupted")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}-interrupted")
        trainer.save_state()

    except Exception as e:
        print(f"\nTraining failed: {e}")
        trainer.save_model(f"{OUTPUT_DIR}-failed")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}-failed")
        trainer.save_state()
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("PHASE 2 PRETRAINING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
