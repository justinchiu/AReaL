#!/usr/bin/env python3
"""
Evaluate Qwen base model on FrontierCO test problems using CO-Bench framework.

This script evaluates the base Qwen 14B model in a single-shot setting
(no iterative refinement) on a subset of FrontierCO problems.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add CO-Bench to path
sys.path.insert(0, str(Path.home() / "CO-Bench"))

from evaluation import YieldingEvaluator, get_new_data


def generate_code(model, tokenizer, problem_description, temperature=0.8, max_tokens=4096):
    """Generate code using the model in a single shot."""

    # Format prompt
    prompt = (
        f"You are an expert in Operations Research and combinatorial optimization. "
        f"Solve the following problem by implementing the required solve function.\n\n"
        f"{problem_description}\n\n"
        f"Important requirements:\n"
        f"1. Your solve function MUST use Python's yield keyword to progressively return better solutions\n"
        f"2. The function should continuously improve solutions over time\n"
        f"3. Enclose all code within ```python ... ``` markers\n"
        f"4. You may use standard Python packages (numpy, scipy, etc.) but avoid commercial solvers\n"
        f"5. The solver will run with a timeout - focus on finding good solutions quickly\n\n"
        f"Generate the complete implementation:"
    )

    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
    ).to(model.device)

    # Generate
    print(f"Generating code... (temperature={temperature})")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            top_p=0.95 if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    return generated


def extract_code_blocks(text):
    """Extract Python code from markdown code blocks."""
    import re

    # Try to find ```python blocks
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return "\n\n".join(matches)

    # Try generic ``` blocks
    pattern = r"```\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return "\n\n".join(matches)

    # Return full text if no blocks found
    return text


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen on FrontierCO")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-14B",
        help="Path to model"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["TSP", "MIS"],
        help="Tasks to evaluate (default: TSP MIS)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/ubuntu/AReaL/data",
        help="Directory containing frontierco data"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for solver execution (seconds)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Generation temperature"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./frontierco_eval_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded successfully!")
    print(f"Evaluating on tasks: {args.tasks}")
    print(f"Timeout: {args.timeout}s")

    all_results = {}

    for task in args.tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        try:
            # Load task data
            data = get_new_data(task, src_dir=args.data_dir, data_dir=args.data_dir)

            # Generate code
            generated_text = generate_code(
                model,
                tokenizer,
                data.problem_description,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )

            # Extract code
            code = extract_code_blocks(generated_text)

            # Save generated code
            code_file = os.path.join(args.output_dir, f"{task}_generated.py")
            with open(code_file, "w") as f:
                f.write(code)
            print(f"Generated code saved to: {code_file}")

            # Evaluate
            print(f"Evaluating with timeout={args.timeout}s...")
            evaluator = YieldingEvaluator(data, timeout=args.timeout)
            feedback = evaluator.evaluate(code)

            # Store results
            all_results[task] = {
                "test_score": feedback.test_score,
                "test_feedback": feedback.test_feedback,
                "dev_score": feedback.dev_score,
                "dev_feedback": feedback.dev_feedback,
            }

            print(f"\nResults for {task}:")
            print(f"  Dev Score: {feedback.dev_score}")
            print(f"  Test Score: {feedback.test_score}")
            print(f"  Test Feedback: {feedback.test_feedback}")

        except Exception as e:
            print(f"Error evaluating {task}: {e}")
            import traceback
            traceback.print_exc()
            all_results[task] = {"error": str(e)}

    # Save results
    results_file = os.path.join(args.output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}")

    # Print summary
    print("\nSummary:")
    for task, result in all_results.items():
        if "error" in result:
            print(f"  {task}: ERROR - {result['error']}")
        else:
            print(f"  {task}: Test Score = {result['test_score']}")


if __name__ == "__main__":
    main()
