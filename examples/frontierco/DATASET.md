# FrontierCo Dataset Configuration

## Overview

The FrontierCo dataset loader is designed to be flexible and work with custom code generation datasets. You need to configure it to point to your actual dataset.

## Quick Setup

### Step 1: Prepare Your Dataset

Your dataset should be in **JSONL format** with the following structure:

```jsonl
{"prompt": "def add(a, b):\n    ", "canonical_solution": "return a + b", "test": "def check(add):\n    assert add(1, 2) == 3\n    assert add(-1, 1) == 0", "entry_point": "add", "task_id": "task_001"}
{"prompt": "def is_even(n):\n    ", "canonical_solution": "return n % 2 == 0", "test": "def check(is_even):\n    assert is_even(2) == True\n    assert is_even(3) == False", "entry_point": "is_even", "task_id": "task_002"}
```

**Required Fields**:
- `prompt`: The coding problem/function signature
- `canonical_solution` (for SFT) or `test` (for RL): The solution or test code
- `entry_point`: Function name being implemented/tested
- `task_id` (optional): Unique identifier

### Step 2: Configure the Dataset Loader

Edit `areal/dataset/frontierco.py` and update the `"frontierco"` section:

```python
elif "frontierco" in path.lower():
    # Option 1: Load from specific file path (recommended)
    dataset = load_dataset('json', data_files='/path/to/your/dataset.jsonl', split='train')

    # Option 2: Load from directory with train/test splits
    # dataset = load_dataset('json', data_dir='/path/to/dataset_dir', split=split)

    # Option 3: Load from HuggingFace Hub (if you uploaded it)
    # dataset = load_dataset("your-username/frontierco", split=split)
```

### Step 3: Update Config Files

If using a file path, you can also update the yaml configs to point directly to your file:

```yaml
train_dataset:
  path: /path/to/your/train.jsonl
  type: sft
  max_length: 8192

valid_dataset:
  path: /path/to/your/test.jsonl
  type: sft
  max_length: 8192
```

## Dataset Format Details

### For SFT (Supervised Fine-Tuning)

**Required columns**:
- `prompt` or `question`: The coding problem
- `canonical_solution` or `code` or `solution`: The reference implementation

**Example**:
```json
{
  "prompt": "def reverse_string(s: str) -> str:\n    \"\"\"Reverse the input string.\"\"\"\n    ",
  "canonical_solution": "return s[::-1]",
  "task_id": "reverse_001"
}
```

### For RL (Reinforcement Learning)

**Required columns**:
- `prompt` or `question`: The coding problem
- `test`: Test code that calls the function with assertions
- `entry_point`: Function name to test
- `task_id` (optional): Unique identifier

**Example**:
```json
{
  "prompt": "def reverse_string(s: str) -> str:\n    \"\"\"Reverse the input string.\"\"\"\n    ",
  "test": "def check(reverse_string):\n    assert reverse_string('hello') == 'olleh'\n    assert reverse_string('') == ''\n    assert reverse_string('a') == 'a'",
  "entry_point": "reverse_string",
  "task_id": "reverse_001"
}
```

## Test Format

The `test` field should contain a function named `check` that takes the implemented function as an argument and runs assertions:

```python
def check(function_name):
    assert function_name(input1) == expected1
    assert function_name(input2) == expected2
    # ... more test cases
```

This format is compatible with HumanEval and makes it easy to verify correctness.

## Example Dataset Creation

Here's a Python script to create a sample dataset:

```python
import json

dataset = [
    {
        "prompt": "def add(a: int, b: int) -> int:\n    \"\"\"Return sum of a and b.\"\"\"\n    ",
        "canonical_solution": "return a + b",
        "test": "def check(add):\n    assert add(1, 2) == 3\n    assert add(0, 0) == 0\n    assert add(-1, 1) == 0",
        "entry_point": "add",
        "task_id": "add_001"
    },
    {
        "prompt": "def multiply(a: int, b: int) -> int:\n    \"\"\"Return product of a and b.\"\"\"\n    ",
        "canonical_solution": "return a * b",
        "test": "def check(multiply):\n    assert multiply(2, 3) == 6\n    assert multiply(0, 5) == 0\n    assert multiply(-2, 3) == -6",
        "entry_point": "multiply",
        "task_id": "multiply_001"
    }
]

# Save as JSONL
with open('frontierco_dataset.jsonl', 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')
```

## Using HumanEval for Testing

For quick testing, you can use HumanEval by changing the path in your configs:

```yaml
train_dataset:
  path: openai/humaneval  # Use HumanEval instead
  type: sft
```

No code changes needed - the loader automatically handles HumanEval format.

## Alternative: Directory Structure

If you have separate train/test files, organize them like this:

```
/path/to/dataset/
├── train.jsonl
└── test.jsonl
```

Then configure:

```python
elif "frontierco" in path.lower():
    dataset = load_dataset('json', data_dir='/path/to/dataset', split=split)
```

And use in configs:

```yaml
train_dataset:
  path: frontierco  # Will load from configured directory
  type: sft
```

## Validation

Before training, validate your dataset:

```python
from datasets import load_dataset

# Test loading
dataset = load_dataset('json', data_files='your_dataset.jsonl', split='train')

# Check columns
print("Columns:", dataset.column_names)

# Check first example
print("Example:", dataset[0])

# Verify required fields exist
required_sft = ['prompt', 'canonical_solution']
required_rl = ['prompt', 'test', 'entry_point']

for field in required_sft:
    assert field in dataset.column_names or any(alt in dataset.column_names for alt in ['question', 'code', 'solution'])
```

## Common Issues

### Issue: "FrontierCo dataset path not configured" error
**Solution**: Edit `areal/dataset/frontierco.py` and uncomment/modify the dataset loading code in the `"frontierco"` section.

### Issue: Missing columns error
**Solution**: Make sure your dataset has the required fields (`prompt`, `canonical_solution` or `test`, etc.)

### Issue: Test execution fails
**Solution**: Verify your `test` field contains valid Python code with a `check()` function that accepts the entry_point function.

## Next Steps

1. Prepare your dataset in JSONL format
2. Update `areal/dataset/frontierco.py` with your dataset path
3. Run a small test to verify loading works
4. Start training!

See `README.md` for training commands.
