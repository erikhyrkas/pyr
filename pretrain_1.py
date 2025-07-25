import os
import gc
from datasets import load_dataset
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

def main():
    print("=" * 70)
    print("PYR PRETRAINING Phase 1")
    print("=" * 70)

    DATASET_SIZE = 20_000_000
    EPOCHS = 2
    MAX_LENGTH = 512
    MAX_POSITION_EMBEDDINGS = 8_192

    print(f"Dataset size: {DATASET_SIZE:,} samples")
    print(f"Training sequence length: {MAX_LENGTH}")
    print(f"Max model capacity: {MAX_POSITION_EMBEDDINGS} tokens")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("./pyr-16k-tokenizer", use_fast=True)

    output_dir = "./pyr-135m-base-1"
    last_checkpoint = get_last_checkpoint(output_dir)

    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        model = LlamaForCausalLM.from_pretrained(last_checkpoint)
    else:
        print("Creating Pyr configuration...")
        config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=768,
            num_hidden_layers=16,
            num_attention_heads=12,
            num_key_value_heads=4,
            intermediate_size=2304,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            hidden_act="relu2",
            tie_word_embeddings=False,
            use_cache=False,
        )
        print("Creating Pyr model (standard transformer)...")
        model = LlamaForCausalLM(config)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Pyr model: {total_params:.1f}M parameters")

    tokenized = build_tokenized_dataset(tokenizer, DATASET_SIZE, MAX_LENGTH)
    gc.collect()
    print("Dataset tokenized")

    batch_size = 16
    grad_accum = 16
    effective_batch_size = batch_size * grad_accum

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=EPOCHS,
        save_total_limit=3,
        logging_steps=50,
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=500,
        bf16=True,
        max_grad_norm=1.0,
        logging_dir=f"{output_dir}/logs",
        logging_first_step=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
        lr_scheduler_type="cosine",
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

    print("\n" + "=" * 70)
    print("STARTING PYR PRETRAINING")
    print("=" * 70)

    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)

        print("\nTRAINING COMPLETED SUCCESSFULLY!")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        trainer.save_state()

        with open(f"{output_dir}/training_info.txt", "w") as f:
            f.write(f"Pyr 135m Base\n")
            f.write(f"Parameters: {total_params:.1f}M\n")
            f.write(f"Training samples: {len(tokenized['train']):,}\n")
            f.write(f"Training sequence length: {MAX_LENGTH}\n")
            f.write(f"Max position embeddings: {MAX_POSITION_EMBEDDINGS}\n")
            f.write(f"Effective batch size: {effective_batch_size}\n")
            f.write(f"Final eval loss: {trainer.state.log_history[-1].get('eval_loss', 'N/A')}\n")
            f.write(f"Architecture: llama with ReLU² activation\n")

        print(f"Model and training info saved to: {output_dir}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        trainer.save_model(f"{output_dir}-interrupted")
        tokenizer.save_pretrained(f"{output_dir}-interrupted")
        trainer.save_state()

    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        trainer.save_model(f"{output_dir}-failed")
        tokenizer.save_pretrained(f"{output_dir}-failed")
        trainer.save_state()

    print("\n" + "=" * 70)
    print("PYR PRETRAINING COMPLETE")
    print("=" * 70)


def get_last_checkpoint(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    checkpoints = [
        os.path.join(output_dir, d) for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    return sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]


def build_tokenized_dataset(tokenizer, DATASET_SIZE, MAX_LENGTH):
    print(f"\nLoading dataset ({DATASET_SIZE:,} samples)...")
    raw = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2")
    train = raw["train"].shuffle(seed=42)
    split = train.select(range(DATASET_SIZE)).train_test_split(test_size=0.01, seed=42)
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
        num_proc=12
    )
    return tokenized


if __name__ == "__main__":
    main()
