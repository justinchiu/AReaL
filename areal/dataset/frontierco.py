"""
FrontierCo dataset for code generation tasks.
Supports SFT and RL training with custom code datasets.

Expected Dataset Format:
    For SFT: Each example should have:
        - 'prompt' or 'question': The coding problem/instruction
        - 'canonical_solution' or 'code' or 'solution': The reference solution

    For RL: Each example should have:
        - 'prompt' or 'question': The coding problem
        - 'test' or 'test_cases': Test code or test case list
        - 'entry_point': Function name being tested
        - 'task_id' (optional): Unique identifier

Example dataset structure (JSONL):
    {"prompt": "def add(a, b):\n    ", "canonical_solution": "return a + b", "test": "assert add(1, 2) == 3", "entry_point": "add"}
"""
from typing import Optional
from datasets import load_dataset


def get_frontierco_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: Optional[int] = None,
):
    """
    Load and format code generation dataset for supervised fine-tuning.

    Applies chat templates and creates loss masks to only train on code generation.

    Args:
        path: Dataset path - can be:
              - HuggingFace dataset: "openai/humaneval"
              - Local directory: "/path/to/dataset"
              - Local file: "/path/to/data.jsonl"
        split: Dataset split ("train" or "test")
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length (filters longer sequences)

    Returns:
        Dataset with input_ids and loss_mask columns

    Expected columns in dataset:
        - 'prompt' or 'question': The coding problem
        - 'canonical_solution' or 'code' or 'solution': The reference code
    """
    # Load dataset based on path type
    if "humaneval" in path.lower():
        # HumanEval doesn't have explicit splits, so we use "test" for all
        dataset = load_dataset(path, split="test")
        if split == "train":
            # Use first 80% for training
            dataset = dataset.select(range(int(len(dataset) * 0.8)))
        else:
            # Use last 20% for validation
            dataset = dataset.select(range(int(len(dataset) * 0.8), len(dataset)))
    elif path.endswith('.jsonl') or path.endswith('.json'):
        # Load from local JSONL/JSON file
        dataset = load_dataset('json', data_files=path, split='train')
    elif "frontierco" in path.lower():
        # Load CardinalOperations/OR-Instruct-Data-3K dataset
        dataset = load_dataset("CardinalOperations/OR-Instruct-Data-3K", split=split)
    else:
        # Try loading as HuggingFace dataset or local directory
        dataset = load_dataset(path, split=split)

    def process(sample):
        """Process a single sample into chat format."""
        # Extract prompt and solution
        if "prompt" in sample:
            prompt = sample["prompt"]
        elif "question" in sample:
            prompt = sample["question"]
        else:
            raise ValueError(f"Unknown prompt key in dataset: {sample.keys()}")

        # Check if this is OR-Instruct format (has 'completion')
        if "completion" in sample:
            # OR-Instruct-Data-3K format: prompt already contains instruction, completion is full solution
            instruction = prompt
            solution = sample["completion"]
        elif "canonical_solution" in sample:
            # HumanEval format: prompt is function signature, need to add instruction
            solution = sample["canonical_solution"]
            instruction = (
                "Complete the following Python function. "
                "Provide only the code inside the function body.\n\n"
                f"{prompt}"
            )
            solution = f"```python\n{solution}\n```"
        elif "code" in sample or "solution" in sample:
            # Generic code format
            solution = sample.get("code") or sample.get("solution")
            instruction = (
                "Complete the following Python function. "
                "Provide only the code inside the function body.\n\n"
                f"{prompt}"
            )
            solution = f"```python\n{solution}\n```"
        else:
            raise ValueError(f"Unknown solution key in dataset: {sample.keys()}")

        # Create messages format with user question and assistant answer
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": solution},
        ]

        # Apply chat template to get the full sequence with special tokens
        seq_token = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,  # We have the assistant response
        )

        # To compute loss mask, we need to know where the assistant response starts
        # Tokenize just the user message to find the prompt length
        user_only = [{"role": "user", "content": instruction}]
        prompt_token = tokenizer.apply_chat_template(
            user_only,
            tokenize=True,
            add_generation_prompt=True,  # Adds <|im_start|>assistant
        )

        # Loss mask: 0 for prompt (user + assistant start), 1 for assistant response
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))

        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process)

    # Remove original columns, keep only input_ids and loss_mask
    original_columns = dataset.column_names
    columns_to_remove = [col for col in original_columns if col not in ["input_ids", "loss_mask"]]
    dataset = dataset.remove_columns(columns_to_remove)

    if max_length is not None:
        # Filter out sequences longer than max_length
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def get_frontierco_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: Optional[int] = None,
):
    """
    Load and format code generation dataset for RL training.

    Returns prompts with test cases for code execution and verification.

    Args:
        path: Dataset path - can be:
              - HuggingFace dataset: "openai/humaneval"
              - Local directory: "/path/to/dataset"
              - Local file: "/path/to/data.jsonl"
        split: Dataset split ("train" or "test")
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length (filters longer prompts)

    Returns:
        Dataset with messages, test_cases, and metadata columns

    Expected columns in dataset:
        - 'prompt' or 'question': The coding problem
        - 'test' or 'test_cases': Test code to run
        - 'entry_point': Function name being tested
        - 'task_id' (optional): Unique identifier
    """
    # Load dataset based on path type
    if "humaneval" in path.lower():
        dataset = load_dataset(path, split="test")
        if split == "train":
            dataset = dataset.select(range(int(len(dataset) * 0.8)))
        else:
            dataset = dataset.select(range(int(len(dataset) * 0.8), len(dataset)))
    elif path.endswith('.jsonl') or path.endswith('.json'):
        # Load from local JSONL/JSON file
        dataset = load_dataset('json', data_files=path, split='train')
    elif "frontierco" in path.lower():
        # Load CardinalOperations/OR-Instruct-Data-3K dataset
        dataset = load_dataset("CardinalOperations/OR-Instruct-Data-3K", split=split)
    else:
        # Try loading as HuggingFace dataset or local directory
        dataset = load_dataset(path, split=split)

    def process(sample):
        """Process a single sample for RL training."""
        # Extract prompt
        if "prompt" in sample:
            prompt = sample["prompt"]
        elif "question" in sample:
            prompt = sample["question"]
        else:
            raise ValueError(f"Unknown prompt key in dataset: {sample.keys()}")

        # Check if this is OR-Instruct format
        if "completion" in sample and "test" not in sample:
            # OR-Instruct-Data-3K format: has prompt/completion but NO test cases
            # For RL, we can use the prompt but rewards won't work without tests
            # Use prompt as-is since it already contains instruction
            instruction = prompt
            test_cases = []  # No test cases available
            entry_point = ""
            task_id = ""
            print("WARNING: OR-Instruct-Data-3K has no test cases. RL training with code execution rewards will not work!")
        else:
            # HumanEval or custom format with test cases
            instruction = (
                "Complete the following Python function. "
                "Provide your solution in a Python code block.\n\n"
                f"{prompt}\n\n"
                "Wrap your solution in ```python``` markers."
            )

            # Extract test cases
            test_cases = []
            if "test" in sample:
                test_code = sample["test"]
                test_cases.append({
                    "type": "code",
                    "test_code": test_code,
                    "entry_point": sample.get("entry_point", ""),
                })

            entry_point = sample.get("entry_point", "")
            task_id = sample.get("task_id", "")

        # Create messages for RL (prompt only, model will generate completion)
        messages = [
            {"role": "user", "content": instruction}
        ]

        return {
            "messages": messages,
            "test_cases": test_cases,
            "entry_point": entry_point,
            "task_id": task_id,
        }

    dataset = dataset.map(process)

    # Remove original columns except the ones we need
    original_columns = dataset.column_names
    columns_to_keep = ["messages", "test_cases", "entry_point", "task_id"]
    columns_to_remove = [col for col in original_columns if col not in columns_to_keep]
    dataset = dataset.remove_columns(columns_to_remove)

    if max_length is not None:
        # Filter out prompts longer than max_length
        def filter_length(sample):
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
