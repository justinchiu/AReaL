#!/usr/bin/env python3
"""
Show sample evaluation problems from OR-Instruct-Data-3K.
"""
from datasets import load_dataset


def show_problems(n_samples=3):
    """Show sample problems from the dataset."""
    print("Loading OR-Instruct-Data-3K dataset...")
    dataset = load_dataset("CardinalOperations/OR-Instruct-Data-3K", split="train")

    print(f"\nDataset size: {len(dataset)} problems")
    print(f"Columns: {dataset.column_names}")
    print("\n" + "="*80)

    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]

        print(f"\n{'='*80}")
        print(f"PROBLEM {i+1}")
        print(f"{'='*80}")

        print("\n--- PROMPT (what the model sees) ---")
        prompt = sample['prompt']
        # Show first 500 chars
        if len(prompt) > 500:
            print(prompt[:500] + "...")
            print(f"\n[Full prompt length: {len(prompt)} characters]")
        else:
            print(prompt)

        print("\n--- COMPLETION (reference solution) ---")
        completion = sample['completion']
        # Show first 800 chars
        if len(completion) > 800:
            print(completion[:800] + "...")
            print(f"\n[Full completion length: {len(completion)} characters]")
        else:
            print(completion)

        print("\n" + "="*80)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    show_problems(n)
