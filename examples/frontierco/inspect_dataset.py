#!/usr/bin/env python3
"""
Quick script to inspect the OR-Instruct-Data-3K dataset format.
"""
from datasets import load_dataset

print("Loading CardinalOperations/OR-Instruct-Data-3K...")
try:
    dataset = load_dataset("CardinalOperations/OR-Instruct-Data-3K", split="train")

    print(f"\nDataset size: {len(dataset)}")
    print(f"\nColumn names: {dataset.column_names}")
    print(f"\nFirst example:")
    print("-" * 80)
    example = dataset[0]
    for key, value in example.items():
        if isinstance(value, str) and len(value) > 200:
            print(f"{key}: {value[:200]}... (truncated)")
        else:
            print(f"{key}: {value}")

    print("\n" + "-" * 80)
    print("\nChecking for test/code related columns...")

    # Check what columns exist
    if 'test' in dataset.column_names:
        print("✓ Has 'test' column")
    if 'test_cases' in dataset.column_names:
        print("✓ Has 'test_cases' column")
    if 'canonical_solution' in dataset.column_names:
        print("✓ Has 'canonical_solution' column")
    if 'code' in dataset.column_names:
        print("✓ Has 'code' column")
    if 'solution' in dataset.column_names:
        print("✓ Has 'solution' column")
    if 'prompt' in dataset.column_names:
        print("✓ Has 'prompt' column")
    if 'question' in dataset.column_names:
        print("✓ Has 'question' column")
    if 'entry_point' in dataset.column_names:
        print("✓ Has 'entry_point' column")

    print("\n" + "-" * 80)
    print("Sample another example:")
    print("-" * 80)
    if len(dataset) > 1:
        example2 = dataset[1]
        for key, value in example2.items():
            if isinstance(value, str) and len(value) > 200:
                print(f"{key}: {value[:200]}... (truncated)")
            else:
                print(f"{key}: {value}")

except Exception as e:
    print(f"Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
