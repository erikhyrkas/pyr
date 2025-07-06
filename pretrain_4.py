import os
import torch
from datasets import load_dataset
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main():
    print("=" * 70)
    print("PRETRAINING Phase 3 - Python EDU")
    print("=" * 70)

    DATASET_SIZE = 7_680_000  # All 7.68M rows from python-edu
    EPOCHS = 1
    MAX_LENGTH = 1024  # Keep same sequence length as Phase 3

    # Path to Phase 3 model
    PHASE3_MODEL_PATH = "./pyr-135m-base-3"  # Adjust if your Phase 3 output dir is different

    print(f"Dataset size: {DATASET_SIZE:,} samples")
    print(f"Training sequence length: {MAX_LENGTH}")
    print(f"Loading from Phase 3 model: {PHASE3_MODEL_PATH}")

    print("\nLoading tokenizer from Phase 3...")
    tokenizer = AutoTokenizer.from_pretrained(PHASE3_MODEL_PATH, use_fast=True)

    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    print(f"\nLoading Pyr model from Phase 3...")
    model = LlamaForCausalLM.from_pretrained(
        PHASE3_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Pyr model loaded: {total_params:.1f}M parameters")

    # Ensure model is on GPU
    if not next(model.parameters()).is_cuda:
        model = model.to("cuda")
        print("Model moved to GPU")

    print(f"\nLoading Python-EDU dataset ({DATASET_SIZE:,} samples)...")
    raw = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu")

    # Use all available data or the specified amount
    total_available = len(raw["train"])
    samples_to_use = min(DATASET_SIZE, total_available)
    print(f"Available samples: {total_available:,}")
    print(f"Using samples: {samples_to_use:,}")

    train = raw["train"].shuffle(seed=42)
    selected_data = train.select(range(samples_to_use))
    split = selected_data.train_test_split(test_size=0.005, seed=42)  # Smaller eval set (0.5%)

    print(f"Dataset loaded:")
    print(f"   Train: {len(split['train']):,} samples")
    print(f"   Eval: {len(split['test']):,} samples")

    def tokenize(batch):
        result = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_attention_mask=True,
            return_tensors=None
        )
        return {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"]
        }

    print("Tokenizing dataset...")
    tokenized = split.map(
        tokenize,
        batched=True,
        remove_columns=split["train"].column_names,
        desc="Tokenizing",
        num_proc=4  # Parallel processing for faster tokenization
    )

    print("Dataset tokenized")

    output_dir = f"./pyr-135m-base-4"

    batch_size = 8  # Same as Phase 3 for 1024 sequences
    grad_accum = 32  # Same effective batch size as Phase 3
    effective_batch_size = batch_size * grad_accum

    print(f"\nTraining configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {grad_accum}")
    print(f"   Effective batch size: {effective_batch_size}")

    # Even lower learning rate for Phase 4 - fine-tuning on specialized data
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=EPOCHS,
        save_total_limit=3,
        logging_steps=100,
        save_steps=2000,  # Less frequent saves due to large dataset
        eval_strategy="steps",
        eval_steps=2000,
        learning_rate=2e-5,  # Lower LR for specialized Python training
        weight_decay=0.01,
        warmup_steps=500,  # More warmup for large dataset
        bf16=True,
        max_grad_norm=1.0,
        logging_dir="./logs-phase4",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
        lr_scheduler_type="cosine",
        # Additional settings for large dataset
        # dataloader_pin_memory=True, # wait to see if we need this
        # gradient_checkpointing=True,  # Save memory at the cost of speed. wait and see if we need this.
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
    )

    total_steps = len(tokenized["train"]) // effective_batch_size * EPOCHS
    print(f"   Total training steps: {total_steps:,}")
    print(f"   Estimated training time: {total_steps * 2.5 / 60:.1f} minutes")

    print("\nTesting forward pass with Python code...")
    sample = tokenized["train"][0]
    test_input = {
        'input_ids': torch.tensor([sample['input_ids'][:512]]).to("cuda"),
        'attention_mask': torch.tensor([sample['attention_mask'][:512]]).to("cuda")
    }

    with torch.no_grad():
        outputs = model(**test_input, labels=test_input['input_ids'])

    print(f"Forward pass successful!")
    print(f"   Current loss: {outputs.loss.item():.4f}")
    print(f"   Output shape: {outputs.logits.shape}")

    # Test with a Python-specific prompt
    print("\nTesting current Python generation capability...")
    test_code = "def fibonacci(n):"
    inputs = tokenizer(test_code, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.3,  # Lower temperature for code
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {test_code}")
    print(f"Generated: {generated_code}")

    print("\n" + "=" * 70)
    print("STARTING PYR PHASE 4 PRETRAINING - PYTHON EDU")
    print("=" * 70)
    print("Press Ctrl+C to stop training and save current checkpoint")

    try:
        trainer.train()

        print("\nPHASE 4 TRAINING COMPLETED SUCCESSFULLY!")
        print("Saving final model...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        with open(f"{output_dir}/training_info.txt", "w") as f:
            f.write(f"Pyr 135m Base - Phase 4 (Python EDU)\n")
            f.write(f"Parameters: {total_params:.1f}M\n")
            f.write(f"Training samples: {len(tokenized['train']):,}\n")
            f.write(f"Training sequence length: {MAX_LENGTH}\n")
            f.write(f"Dataset: Python-EDU (HuggingFaceTB/smollm-corpus)\n")
            f.write(f"Effective batch size: {effective_batch_size}\n")
            f.write(f"Final eval loss: {trainer.state.log_history[-1].get('eval_loss', 'N/A')}\n")
            f.write(f"Continued from: {PHASE3_MODEL_PATH}\n")
            f.write(f"Learning rate: {training_args.learning_rate}\n")
            f.write(f"Specialization: Python programming and education\n")
            f.write(f"Architecture: BitNet with ReLU² activation\n")

        print(f"Model saved to: {output_dir}")
        print(f"Training info saved to: {output_dir}/training_info.txt")

        print("\nTesting final Python generation capability...")
        test_prompts = [
            "def quicksort(arr):",
            "# Calculate the factorial of a number\ndef factorial(n):",
            "class LinkedList:",
            "import numpy as np\n\n# Function to calculate mean"
        ]

        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated}")
            print("-" * 50)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving current checkpoint...")
        model.save_pretrained(f"{output_dir}-interrupted")
        tokenizer.save_pretrained(f"{output_dir}-interrupted")
        print(f"Checkpoint saved to: {output_dir}-interrupted")

    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("PYR PHASE 4 PRETRAINING COMPLETE")
    print("=" * 70)
    print("Ready for instruction tuning (Phase 5)!")


if __name__ == "__main__":
    main()