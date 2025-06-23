import os
import torch
from datasets import load_dataset
from transformers import (
    BitNetForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


def main():
    print("=" * 70)
    print("PRETRAINING Phase 2")
    print("=" * 70)

    DATASET_SIZE = 10_000_000  # 10M samples = ~10B tokens for length 1024
    EPOCHS = 1
    MAX_LENGTH = 1024  # Increased sequence length for phase 2

    # Path to Phase 1 model
    PHASE1_MODEL_PATH = "./pyr-135m-base-1"

    print(f"Dataset size: {DATASET_SIZE:,} samples")
    print(f"Training sequence length: {MAX_LENGTH}")
    print(f"Loading from Phase 1 model: {PHASE1_MODEL_PATH}")

    print("\nLoading tokenizer from Phase 1...")
    tokenizer = AutoTokenizer.from_pretrained(PHASE1_MODEL_PATH, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    print(f"\nLoading Pyr model from Phase 1...")
    model = BitNetForCausalLM.from_pretrained(
        PHASE1_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Pyr model loaded: {total_params:.1f}M parameters")

    # Ensure model is on GPU
    if not next(model.parameters()).is_cuda:
        model = model.to("cuda")
        print("Model moved to GPU")

    print(f"\nLoading FineWeb-EDU dataset ({DATASET_SIZE:,} samples)...")
    raw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT")

    # Take a subset and split for training
    selected_data = raw["train"].select(range(min(DATASET_SIZE, len(raw["train"]))))
    split = selected_data.train_test_split(test_size=0.01, seed=42)

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

    output_dir = f"./pyr-135m-base-2"

    batch_size = 8  # Reduced due to longer sequences (1024 vs 512)
    grad_accum = 32  # Increased to maintain similar effective batch size
    effective_batch_size = batch_size * grad_accum

    print(f"\nTraining configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {grad_accum}")
    print(f"   Effective batch size: {effective_batch_size}")

    # Lower learning rate for continued training
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=EPOCHS,
        save_total_limit=3,
        logging_steps=100,
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        learning_rate=5e-5,  # Lower LR for continued training
        weight_decay=0.01,
        warmup_steps=250,  # Fewer warmup steps since continuing training
        fp16=True,
        max_grad_norm=1.0,
        logging_dir="./logs-phase2",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
        lr_scheduler_type="cosine",  # Cosine decay for phase 2
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
    print(f"   Estimated training time: {total_steps * 2.5 / 60:.1f} minutes")  # Longer due to 1024 seq len

    print("\nTesting forward pass with longer sequence...")
    sample = tokenized["train"][0]
    test_input = {
        'input_ids': torch.tensor([sample['input_ids'][:512]]).to("cuda"),  # Test with 512 first
        'attention_mask': torch.tensor([sample['attention_mask'][:512]]).to("cuda")
    }

    with torch.no_grad():
        outputs = model(**test_input, labels=test_input['input_ids'])

    print(f"Forward pass successful!")
    print(f"   Current loss: {outputs.loss.item():.4f}")
    print(f"   Output shape: {outputs.logits.shape}")

    print("\n" + "=" * 70)
    print("STARTING PYR PHASE 2 PRETRAINING")
    print("=" * 70)
    print("Press Ctrl+C to stop training and save current checkpoint")

    try:
        trainer.train()

        print("\nPHASE 2 TRAINING COMPLETED SUCCESSFULLY!")
        print("Saving final model...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        with open(f"{output_dir}/training_info.txt", "w") as f:
            f.write(f"Pyr 135m Base - Phase 2\n")
            f.write(f"Parameters: {total_params:.1f}M\n")
            f.write(f"Training samples: {len(tokenized['train']):,}\n")
            f.write(f"Training sequence length: {MAX_LENGTH}\n")
            f.write(f"Dataset: FineWeb-EDU\n")
            f.write(f"Effective batch size: {effective_batch_size}\n")
            f.write(f"Final eval loss: {trainer.state.log_history[-1].get('eval_loss', 'N/A')}\n")
            f.write(f"Continued from: {PHASE1_MODEL_PATH}\n")
            f.write(f"Learning rate: {training_args.learning_rate}\n")
            f.write(f"Architecture: BitNet with ReLU² activation\n")

        print(f"Model saved to: {output_dir}")
        print(f"Training info saved to: {output_dir}/training_info.txt")

        print("\nTesting trained model...")
        test_text = "Machine learning is a subset of artificial intelligence that"
        inputs = tokenizer(test_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {test_text}")
        print(f"Generated: {generated_text}")

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
    print("PYR PHASE 2 PRETRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
