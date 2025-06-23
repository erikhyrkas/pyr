import time

from transformers import PreTrainedTokenizerFast, AutoTokenizer


def load_tokenizers():
    """Load all tokenizers for comparison"""
    tokenizers = {}

    # Pyr tokenizer (assumes it's in ./pyr-tokenizer)
    try:
        tokenizers["Pyr-16K"] = PreTrainedTokenizerFast.from_pretrained("./pyr-16k-tokenizer")
        print("✅ Loaded Pyr-16K tokenizer")
    except Exception as e:
        print(f"❌ Failed to load Pyr tokenizer: {e}")
        return None

    # Reference tokenizers
    reference_tokenizers = {
        "GPT-2": "gpt2",
        "SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M",
        "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",  # If available
    }

    for name, model_name in reference_tokenizers.items():
        try:
            tokenizers[name] = AutoTokenizer.from_pretrained(model_name)
            print(f"✅ Loaded {name} tokenizer")
        except Exception as e:
            print(f"⚠️  Failed to load {name}: {e}")

    return tokenizers


def test_basic_words(tokenizers):
    """Test common English words"""
    print("\n" + "=" * 80)
    print("BASIC WORD TOKENIZATION TEST")
    print("=" * 80)

    basic_words = [
        "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "has", "have", "had", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "will", "would", "could", "should", "can", "may",
        "this", "that", "these", "those", "what", "which", "who", "when", "where",
        "how", "why", "because", "if", "then", "else", "while", "for", "each"
    ]

    for word in basic_words:
        print(f"\nWord: '{word}'")
        for name, tokenizer in tokenizers.items():
            tokens = tokenizer.encode(word, add_special_tokens=False)
            decoded_tokens = [tokenizer.decode([token]) for token in tokens]
            status = "✅" if len(tokens) == 1 else "❌"
            print(f"  {name:15}: {status} {len(tokens)} tokens {decoded_tokens}")


def test_programming_terms(tokenizers):
    """Test programming-related vocabulary"""
    print("\n" + "=" * 80)
    print("PROGRAMMING VOCABULARY TEST")
    print("=" * 80)

    programming_terms = [
        "def", "class", "import", "from", "return", "if", "else", "for", "while",
        "try", "except", "with", "as", "lambda", "yield", "async", "await",
        "True", "False", "None", "self", "cls", "function", "method", "variable",
        "list", "dict", "tuple", "set", "string", "integer", "float", "boolean",
        "print", "input", "len", "range", "enumerate", "zip", "map", "filter",
        "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "//", "**", "->", "=>",
        "python", "javascript", "typescript", "react", "nodejs", "numpy", "pandas"
    ]

    single_token_count = {name: 0 for name in tokenizers.keys()}

    for term in programming_terms:
        print(f"\nTerm: '{term}'")
        for name, tokenizer in tokenizers.items():
            tokens = tokenizer.encode(term, add_special_tokens=False)
            decoded_tokens = [tokenizer.decode([token]) for token in tokens]
            status = "✅" if len(tokens) == 1 else "❌"
            if len(tokens) == 1:
                single_token_count[name] += 1
            print(f"  {name:15}: {status} {len(tokens)} tokens {decoded_tokens}")

    print(f"\n📊 PROGRAMMING TERMS SUMMARY:")
    total_terms = len(programming_terms)
    for name, count in single_token_count.items():
        percentage = (count / total_terms) * 100
        print(f"  {name:15}: {count:2d}/{total_terms} ({percentage:5.1f}%) as single tokens")


def test_numbers_and_math(tokenizers):
    """Test numerical handling"""
    print("\n" + "=" * 80)
    print("NUMBERS AND MATHEMATICS TEST")
    print("=" * 80)

    number_tests = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",  # Individual digits
        "10", "42", "123", "1024", "2024", "9999",  # Multi-digit numbers
        "3.14", "98.6", "0.001", "123.456",  # Decimals
        "1,000", "1,000,000", "123,456,789",  # Comma-separated
        "1+1", "2*3", "10/5", "2**8", "x==42",  # Math expressions
        "Calculate 123 + 456 = 579",  # Math in context
        "The year 2024 has 365 days",  # Numbers in sentences
        "Model has 135300000 parameters",  # Large numbers
    ]

    for test in number_tests:
        print(f"\nText: '{test}'")
        for name, tokenizer in tokenizers.items():
            tokens = tokenizer.encode(test, add_special_tokens=False)
            decoded_tokens = [tokenizer.decode([token]) for token in tokens]
            print(f"  {name:15}: {len(tokens):2d} tokens {decoded_tokens}")


def test_instruction_following(tokenizers):
    """Test instruction-following phrases"""
    print("\n" + "=" * 80)
    print("INSTRUCTION FOLLOWING PHRASES TEST")
    print("=" * 80)

    instruction_phrases = [
        "Please", "Thank you", "Could you", "Can you", "Would you",
        "How do I", "What is", "Why does", "When will", "Where can",
        "I need", "I want", "I would like", "Help me", "Show me",
        "Explain", "Describe", "Compare", "List", "Summarize",
        "Generate", "Create", "Build", "Make", "Write",
        "Please help me with", "Can you explain how",
        "I need you to", "Could you please",
    ]

    for phrase in instruction_phrases:
        print(f"\nPhrase: '{phrase}'")
        for name, tokenizer in tokenizers.items():
            tokens = tokenizer.encode(phrase, add_special_tokens=False)
            decoded_tokens = [tokenizer.decode([token]) for token in tokens]
            print(f"  {name:15}: {len(tokens):2d} tokens {decoded_tokens}")


def test_code_samples(tokenizers):
    """Test complete code samples"""
    print("\n" + "=" * 80)
    print("CODE SAMPLE TOKENIZATION TEST")
    print("=" * 80)

    code_samples = [
        "def hello_world():\n    return 'Hello, World!'",
        "if x == 42:\n    print('Found the answer!')",
        "for i in range(10):\n    print(i)",
        "import numpy as np\narray = np.zeros(100)",
        "class MyClass:\n    def __init__(self):\n        self.value = 0",
        "try:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print('Error!')",
        "lambda x: x * 2",
        "async def fetch_data():\n    return await api_call()",
    ]

    for i, code in enumerate(code_samples, 1):
        print(f"\nCode Sample {i}:")
        print(f"```\n{code}\n```")
        for name, tokenizer in tokenizers.items():
            tokens = tokenizer.encode(code, add_special_tokens=False)
            print(f"  {name:15}: {len(tokens):2d} tokens")


def test_chatml_conversations(tokenizers):
    """Test ChatML conversation formatting"""
    print("\n" + "=" * 80)
    print("CHATML CONVERSATION TEST")
    print("=" * 80)

    # Test if tokenizers have chat templates
    conversations = [
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."}
        ],
        [
            {"role": "user", "content": "Write a Python function to calculate factorial."},
            {"role": "assistant",
             "content": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"}
        ]
    ]

    for i, conversation in enumerate(conversations, 1):
        print(f"\nConversation {i}:")
        for message in conversation:
            print(f"  {message['role']}: {message['content'][:50]}...")

        for name, tokenizer in tokenizers.items():
            try:
                if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                    formatted = tokenizer.apply_chat_template(conversation, tokenize=False)
                    tokens = tokenizer.encode(formatted, add_special_tokens=False)
                    print(f"  {name:15}: {len(tokens):3d} tokens (with chat template)")
                else:
                    # Manual formatting for tokenizers without chat templates
                    manual_format = ""
                    for msg in conversation:
                        manual_format += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                    tokens = tokenizer.encode(manual_format, add_special_tokens=False)
                    print(f"  {name:15}: {len(tokens):3d} tokens (manual ChatML)")
            except Exception as e:
                print(f"  {name:15}: Error processing conversation: {e}")


def benchmark_speed(tokenizers):
    """Benchmark tokenization speed"""
    test_text = "The quick brown fox jumps over the lazy dog. " * 100

    for name, tokenizer in tokenizers.items():
        start_time = time.time()
        for _ in range(100):
            tokenizer.encode(test_text, add_special_tokens=False)
        end_time = time.time()

        tokens_per_second = (100 * len(tokenizer.encode(test_text))) / (end_time - start_time)
        print(f"{name:15}: {tokens_per_second:,.0f} tokens/second")


def test_special_tokens(tokenizers):
    """Test handling of special tokens and edge cases"""
    special_cases = [
        "",  # Empty string
        " ",  # Single space
        "\n",  # Newline
        "\t",  # Tab
        "    ",  # Multiple spaces
        "<|endoftext|>",  # Common special token
        "[MASK]",  # BERT-style mask
        "<unk>",  # Unknown token
    ]

    for case in special_cases:
        print(f"\nSpecial case: {repr(case)}")
        for name, tokenizer in tokenizers.items():
            tokens = tokenizer.encode(case, add_special_tokens=False)
            print(f"  {name:15}: {len(tokens)} tokens")


def test_efficiency_metrics(tokenizers):
    """Calculate overall efficiency metrics"""
    print("\n" + "=" * 80)
    print("TOKENIZER EFFICIENCY METRICS")
    print("=" * 80)

    # Test corpus covering various use cases
    test_corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Please help me debug this Python code that's not working properly.",
        "def calculate_average(numbers): return sum(numbers) / len(numbers)",
        "I need to process 1,234,567 records and it's taking too long.",
        "Can you explain how machine learning models work?",
        "import tensorflow as tf\nmodel = tf.keras.Sequential()",
        "The temperature today is 72.5 degrees Fahrenheit.",
        "Thank you for your help with this complex problem!",
        "if user_input == 'quit': break",
        "What's the difference between list and tuple in Python?",
    ]

    print(f"Test corpus: {len(test_corpus)} diverse sentences\n")

    results = {}
    for name, tokenizer in tokenizers.items():
        total_chars = sum(len(text) for text in test_corpus)
        total_tokens = sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in test_corpus)

        results[name] = {
            "total_tokens": total_tokens,
            "total_chars": total_chars,
            "chars_per_token": total_chars / total_tokens,
            "compression_ratio": total_chars / total_tokens,
            "vocab_size": tokenizer.vocab_size,
        }

    # Sort by efficiency (fewer tokens = better compression)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["total_tokens"])

    print("📊 EFFICIENCY RANKING (lower tokens = better compression):")
    print(f"{'Tokenizer':<15} {'Tokens':<8} {'Chars/Token':<12} {'Vocab Size':<12} {'Efficiency'}")
    print("-" * 70)

    for name, metrics in sorted_results:
        efficiency = "🏆" if metrics["total_tokens"] == min(r[1]["total_tokens"] for r in results.items()) else ""
        print(
            f"{name:<15} {metrics['total_tokens']:<8} {metrics['chars_per_token']:<12.2f} {metrics['vocab_size']:<12,} {efficiency}")


def generate_report(tokenizers):
    """Generate a summary report"""
    print("\n" + "=" * 80)
    print("TOKENIZER COMPARISON SUMMARY REPORT")
    print("=" * 80)

    print("\n🎯 KEY FINDINGS:")
    print("• Programming vocabulary tokenization")
    print("• Common English word handling")
    print("• Numerical reasoning capabilities")
    print("• Instruction-following efficiency")
    print("• Overall compression performance")

    print("\n📋 RECOMMENDATIONS:")
    print("• Check if common words like 'has', 'the', 'and' are single tokens")
    print("• Verify programming keywords are efficiently tokenized")
    print("• Ensure digit splitting works for numerical reasoning")
    print("• Test ChatML format compatibility")
    print("• Compare compression efficiency vs. vocabulary size")

    print(f"\n📈 TOKENIZERS TESTED: {len(tokenizers)}")
    for name, tokenizer in tokenizers.items():
        print(f"  • {name}: {tokenizer.vocab_size:,} vocabulary size")


def main():
    """Run comprehensive tokenizer testing"""
    print("🔍 COMPREHENSIVE TOKENIZER TESTING SUITE")
    print("=" * 80)

    tokenizers = load_tokenizers()
    if not tokenizers:
        print("❌ No tokenizers loaded. Exiting.")
        return

    # Run all test suites
    test_basic_words(tokenizers)
    test_programming_terms(tokenizers)
    test_numbers_and_math(tokenizers)
    test_instruction_following(tokenizers)
    test_code_samples(tokenizers)
    test_chatml_conversations(tokenizers)
    test_special_tokens(tokenizers)
    test_efficiency_metrics(tokenizers)
    benchmark_speed(tokenizers)
    generate_report(tokenizers)

    print("\n✅ TESTING COMPLETE!")
    print("Check the results above to evaluate your tokenizer's performance.")


if __name__ == "__main__":
    main()
