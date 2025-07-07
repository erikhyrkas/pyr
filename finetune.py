import os
import torch
from datasets import load_dataset
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from typing import Dict, List, Any

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ChatDataCollator:
    """Custom data collator for chat data with dynamic padding and left padding."""

    def __init__(self, tokenizer, max_length=8192):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract input_ids and find the maximum length in this batch
        input_ids = [f["input_ids"] for f in features]
        batch_max_length = min(max(len(ids) for ids in input_ids), self.max_length)

        # Left pad all sequences to batch_max_length
        padded_input_ids = []
        attention_masks = []

        for ids in input_ids:
            # Truncate if too long
            if len(ids) > self.max_length:
                ids = ids[-self.max_length:]  # Keep the end (most recent conversation)

            # Calculate padding needed
            padding_length = batch_max_length - len(ids)

            # Left pad with pad_token_id
            padded_ids = [self.tokenizer.pad_token_id] * padding_length + ids

            # Create attention mask (0 for padding, 1 for real tokens)
            attention_mask = [0] * padding_length + [1] * len(ids)

            padded_input_ids.append(padded_ids)
            attention_masks.append(attention_mask)

        # Convert to tensors
        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }

        # For causal LM, labels are the same as input_ids
        # but we mask out the padding tokens in the loss
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["attention_mask"] == 0] = -100

        return batch


def format_conversation(conversation: List[Dict[str, str]], tokenizer) -> str:
    """Format a conversation using the ChatML template."""
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        return tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    else:
        # Fallback to manual ChatML formatting
        formatted = ""
        for message in conversation:
            formatted += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        return formatted


def main():
    print("=" * 70)
    print("PYR INSTRUCTION FINE-TUNING")
    print("=" * 70)

    # Configuration
    DATASET_SIZE = 460_341  # Use all available smol-smoltalk data
    EPOCHS = 3
    MAX_LENGTH = 8_192  # Maximum sequence length

    # Path to your base model (adjust as needed)
    BASE_MODEL_PATH = "./pyr-135m-base-3"  # or "./pyr-135m-base-2" if phase 3 isn't complete

    print(f"Dataset size: {DATASET_SIZE:,} conversations")
    print(f"Max sequence length: {MAX_LENGTH}")
    print(f"Loading from base model: {BASE_MODEL_PATH}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)

    # Set padding side to left for generation tasks
    tokenizer.padding_side = "left"

    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    print(f"Padding side: {tokenizer.padding_side}")
    print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    # Load model
    print(f"\nLoading Pyr model from {BASE_MODEL_PATH}...")
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {total_params:.1f}M parameters")

    # Ensure model is on GPU
    if not next(model.parameters()).is_cuda:
        model = model.to("cuda")
        print("Model moved to GPU")

    # Load and prepare dataset
    print(f"\nLoading smol-smoltalk dataset...")
    raw_dataset = load_dataset("HuggingFaceTB/smol-smoltalk")

    # Take subset if needed
    total_available = len(raw_dataset["train"])
    samples_to_use = min(DATASET_SIZE, total_available)
    print(f"Available conversations: {total_available:,}")
    print(f"Using conversations: {samples_to_use:,}")

    if samples_to_use < total_available:
        train = raw_dataset["train"].shuffle(seed=42)
        dataset = train.select(range(samples_to_use))
    else:
        dataset = raw_dataset["train"]

    # Split for training and evaluation
    split_dataset = dataset.train_test_split(test_size=0.01, seed=42)

    print(f"Dataset split:")
    print(f"   Train: {len(split_dataset['train']):,} conversations")
    print(f"   Eval: {len(split_dataset['test']):,} conversations")

    def process_conversation(example):
        """Process a single conversation example."""
        conversation = example["messages"]

        # Format using ChatML
        formatted_text = format_conversation(conversation, tokenizer)

        # Tokenize
        encoded = tokenizer(
            formatted_text,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=False,  # ChatML template already includes special tokens
            return_tensors=None
        )

        return {
            "input_ids": encoded["input_ids"],
            "length": len(encoded["input_ids"])  # Keep track of length for analysis
        }

    print("\nProcessing conversations...")
    processed_dataset = split_dataset.map(
        process_conversation,
        remove_columns=split_dataset["train"].column_names,
        desc="Processing conversations",
        num_proc=4
    )

    # Analyze length distribution
    train_lengths = [ex["length"] for ex in processed_dataset["train"]]
    print(f"\nLength statistics:")
    print(f"   Min length: {min(train_lengths)} tokens")
    print(f"   Max length: {max(train_lengths)} tokens")
    print(f"   Average length: {sum(train_lengths) / len(train_lengths):.1f} tokens")
    print(f"   Sequences > 4096 tokens: {sum(1 for l in train_lengths if l > 4096)}")
    print(f"   Sequences > 6144 tokens: {sum(1 for l in train_lengths if l > 6144)}")
    print(f"   Sequences at max length: {sum(1 for l in train_lengths if l == MAX_LENGTH)}")

    # Training configuration
    output_dir = "./pyr-135m-instruct"

    # Adjust batch size based on memory constraints
    batch_size = 4  # Start conservative due to variable lengths
    grad_accum = 64  # Higher accumulation to maintain effective batch size
    effective_batch_size = batch_size * grad_accum

    print(f"\nTraining configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {grad_accum}")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Epochs: {EPOCHS}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=EPOCHS,
        save_total_limit=3,
        logging_steps=50,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        learning_rate=2e-5,  # Lower learning rate for instruction tuning
        weight_decay=0.01,
        warmup_steps=100,
        bf16=True,
        max_grad_norm=1.0,
        logging_dir="./logs-instruct",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
        lr_scheduler_type="cosine",
        # gradient_checkpointing=True,  # Save memory at the cost of speed. wait and see if we need this.
        # dataloader_pin_memory=True, # wait to see if we need this
        group_by_length=True,  # Group similar lengths together for efficiency
    )

    # Create custom data collator
    data_collator = ChatDataCollator(tokenizer, max_length=MAX_LENGTH)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
        data_collator=data_collator,
    )

    total_steps = len(processed_dataset["train"]) // effective_batch_size * EPOCHS
    print(f"   Total training steps: {total_steps:,}")
    print(f"   Estimated training time: {total_steps * 3.0 / 60:.1f} minutes")

    # Test forward pass
    print("\nTesting forward pass with conversation data...")
    sample_batch = [processed_dataset["train"][i] for i in range(min(2, len(processed_dataset["train"])))]
    test_batch = data_collator(sample_batch)

    # Move to device
    test_batch = {k: v.to(model.device) for k, v in test_batch.items()}

    with torch.no_grad():
        outputs = model(**test_batch)

    print(f"Forward pass successful!")
    print(f"   Batch size: {test_batch['input_ids'].shape[0]}")
    print(f"   Sequence length: {test_batch['input_ids'].shape[1]}")
    print(f"   Current loss: {outputs.loss.item():.4f}")

    # Test current conversation ability
    print("\nTesting current conversation capability...")
    test_conversation = [
        {"role": "user", "content": "Hello! Can you help me with a Python question?"},
    ]

    formatted_input = format_conversation(test_conversation, tokenizer)
    inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input conversation: {formatted_input[:100]}...")
    print(f"Generated response: {generated_text[len(formatted_input):]}")

    print("\n" + "=" * 70)
    print("STARTING INSTRUCTION FINE-TUNING")
    print("=" * 70)
    print("Press Ctrl+C to stop training and save current checkpoint")

    try:
        trainer.train()

        print("\nINSTRUCTION FINE-TUNING COMPLETED SUCCESSFULLY!")
        print("Saving final model...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save training info
        with open(f"{output_dir}/training_info.txt", "w") as f:
            f.write(f"Pyr 135m Instruct\n")
            f.write(f"Parameters: {total_params:.1f}M\n")
            f.write(f"Training conversations: {len(processed_dataset['train']):,}\n")
            f.write(f"Max sequence length: {MAX_LENGTH}\n")
            f.write(f"Dataset: smol-smoltalk\n")
            f.write(f"Effective batch size: {effective_batch_size}\n")
            f.write(f"Final eval loss: {trainer.state.log_history[-1].get('eval_loss', 'N/A')}\n")
            f.write(f"Base model: {BASE_MODEL_PATH}\n")
            f.write(f"Learning rate: {training_args.learning_rate}\n")
            f.write(f"Training type: Instruction fine-tuning\n")
            f.write(f"Chat format: ChatML\n")
            f.write(f"Padding strategy: Left padding with dynamic batch sizing\n")

        print(f"Model saved to: {output_dir}")
        print(f"Training info saved to: {output_dir}/training_info.txt")

        # Test final conversation ability
        print("\nTesting final conversation capability...")
        test_conversations = [
            [{"role": "user", "content": "Write a simple Python function to calculate fibonacci numbers."}],
            [{"role": "user", "content": "Explain what machine learning is in simple terms."}],
            [{"role": "user", "content": "How do I debug a Python script that's not working?"}],
        ]

        for i, conversation in enumerate(test_conversations, 1):
            formatted_input = format_conversation(conversation, tokenizer)
            inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(formatted_input):]

            print(f"\nTest {i}:")
            print(f"User: {conversation[0]['content']}")
            print(f"Assistant: {response.strip()}")
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
    print("INSTRUCTION FINE-TUNING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()